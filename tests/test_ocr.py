import copy
import io
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lecture2doc.ocr import OCRConfig, RapidEngine, annotate_record, recognize_pages
from lecture2doc.server import create_app
from lecture2doc.storage import digest, read, write


class Engine:
    identity = {"engine": "test-fixture", "model_hash": "test-only"}

    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        assert image[0, 0, 0] == 180  # B, never A.
        return [dict(polygon=[[2, 3], [20, 3], [20, 12], [2, 12]], text="실제 원문 <b>", score=.6),
                dict(polygon=[[22, 3], [30, 3], [30, 12], [22, 12]], text="low", score=.2)]


def record(root):
    root.mkdir()
    a = np.full((80, 100, 3), 50, np.uint8)
    b = np.full_like(a, 180)
    cv2.imwrite(str(root / "a.png"), a)
    cv2.imwrite(str(root / "b.png"), b)
    manifest = dict(source={"path": "a.png", "name": "test"}, playback="a.png",
                    source_frames=[dict(id="f1", image="a.png", time=7.25)],
                    config=dict(roi=[.1, .25, .8, .5]), board_states=[], reconstructed_boards=[],
                    board_pages=[dict(id="p1", selected_frame_id="f1")], page_reconstruction_status="complete",
                    page_reconstructions=[dict(board_page_id="p1", image="b.png", source_frame_ids=["f1"], provenance=None)])
    write(root / "manifest.json", manifest)
    return manifest


def test_native_coordinates_B_input_and_cache(tmp_path):
    m = record(tmp_path / "record")
    before = copy.deepcopy(m)
    engine = Engine()
    result = recognize_pages(tmp_path / "record", m, engine=engine)
    assert result["cache_hits"] == 0 and result["inference_pages"] == 1
    assert result["elapsed_seconds"] >= 0
    page = result["pages"][0]
    assert page["input"]["output"] == "B"
    assert page["input"]["sha256"] == digest(tmp_path / "record/b.png")
    assert page["regions"][0]["bbox"] == [12, 23, 19, 10]
    assert page["regions"][0]["review"]
    assert page["text"] == "실제 원문 <b>"
    patch = cv2.imread(str(tmp_path / "record" / page["regions"][0]["image"]))
    assert patch.shape == (10, 19, 3) and (patch == 180).all()
    assert m == before
    cached = recognize_pages(tmp_path / "record", m, engine=engine)
    assert cached["cache_hits"] == 1 and cached["inference_pages"] == 0
    assert engine.calls == 1
    forced = recognize_pages(tmp_path / "record", m, force=True, engine=engine)
    assert forced["pages"][0]["json_file"] != page["json_file"]
    assert (tmp_path / "record" / page["json_file"]).exists()
    m["config"]["roi"] = [.2, .25, .7, .5]
    recognize_pages(tmp_path / "record", m, engine=engine)
    assert engine.calls == 3


def test_missing_models_and_language(tmp_path):
    with pytest.raises(FileNotFoundError):
        RapidEngine(OCRConfig(), directory=tmp_path)
    with pytest.raises(ValueError):
        OCRConfig(language="auto").validate()


def test_artifacts_allowed_but_unlisted_file_blocked(tmp_path):
    root = tmp_path / "record"
    m = record(root)
    m["ocr"] = recognize_pages(root, m, engine=Engine())
    write(root / "manifest.json", m)
    (root / "private.txt").write_text("private")
    with TestClient(create_app(tmp_path / "data", root)) as client:
        base = "/api/jobs/000000000000000000000001/files/"
        for artifact in m["ocr"]["pages"][0]["artifacts"]:
            assert client.get(base + artifact).status_code == 200
        assert client.get(base + "private.txt").status_code == 404


def test_snapshot_preserves_baseline(tmp_path, monkeypatch):
    from lecture2doc import ocr
    root = tmp_path / "before"
    record(root)
    original = (root / "manifest.json").read_bytes()
    actual = ocr.recognize_pages
    monkeypatch.setattr(ocr, "recognize_pages", lambda root, m, config, progress, force:
                        actual(root, m, config, progress, force, engine=Engine()))
    result = annotate_record(root, output=tmp_path / "after")
    assert result["product_version"] == "V1"
    assert (root / "manifest.json").read_bytes() == original
    assert "ocr" not in read(root / "manifest.json")
    assert read(tmp_path / "after/manifest.json")["ocr"]["status"] == "complete"
    with pytest.raises(ValueError):
        annotate_record(root, output=tmp_path / "after")



def test_prepare_korean_models_downloads_handwriting_recognizer(tmp_path, monkeypatch):
    from lecture2doc import ocr

    calls = []

    class FakeFileInfo:
        def __init__(self, *args):
            self.args = args

    class FakeInferSession:
        @staticmethod
        def get_model_url(choice):
            task = choice.args[2]
            if task == "det":
                return {"model_dir": "https://models.example/ppocrv6-det.onnx"}
            return {
                "model_dir": "https://models.example/korean_PP-OCRv5_mobile_rec.onnx",
                "dict_url": "https://models.example/korean_dict.txt",
            }

    rapidocr = types.ModuleType("rapidocr")
    rapidocr.__path__ = []
    inference_engine = types.ModuleType("rapidocr.inference_engine")
    inference_engine.__path__ = []
    base = types.ModuleType("rapidocr.inference_engine.base")
    base.FileInfo = FakeFileInfo
    base.InferSession = FakeInferSession
    utils = types.ModuleType("rapidocr.utils")
    utils.__path__ = []
    typings = types.ModuleType("rapidocr.utils.typings")
    typings.EngineType = types.SimpleNamespace(ONNXRUNTIME="onnxruntime")
    typings.LangDet = types.SimpleNamespace(CH="ch")
    typings.LangRec = types.SimpleNamespace(KOREAN="korean", EN="en")
    typings.ModelType = types.SimpleNamespace(SMALL="small", MOBILE="mobile", MEDIUM="medium")
    typings.OCRVersion = types.SimpleNamespace(PPOCRV5="v5", PPOCRV6="v6")
    typings.TaskType = types.SimpleNamespace(DET="det", REC="rec")

    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine", inference_engine)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine.base", base)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", typings)
    monkeypatch.setattr(ocr, "version", lambda _name: "3.10.0")

    def fake_urlopen(url, timeout=60):
        calls.append(url)
        return io.BytesIO(url.encode("utf-8"))

    monkeypatch.setattr(ocr.urllib.request, "urlopen", fake_urlopen)

    result = ocr.prepare_models("ko", directory=tmp_path)

    assert result["models"]["rec"] == "korean-PP-OCRv5-mobile"
    assert result["model_sources"]["rec"]["supported_languages"] == ["Korean", "English"]
    assert result["experimental_models"]["rec"] == "korean-handwriting-rec"
    assert result["files"]["rec"]["path"] == "korean_PP-OCRv5_mobile_rec.onnx"
    assert result["files"]["dictionary"]["path"] == "korean_dict.txt"
    assert result["files"]["rec_handwriting"]["path"] == "rec_korean_hw.onnx"
    assert result["files"]["dictionary_handwriting"]["path"] == "ppocrv5_korean_dict.txt"
    assert ocr.KOREAN_HANDWRITING_MODEL_FILES["rec"] in calls
    assert ocr.KOREAN_HANDWRITING_MODEL_FILES["dictionary"] in calls
    assert (tmp_path / "ko.json").is_file()
    assert (tmp_path / "rec_korean_hw.onnx").is_file()
    assert (tmp_path / "ppocrv5_korean_dict.txt").is_file()
    assert (tmp_path / "korean_PP-OCRv5_mobile_rec.onnx").is_file()


def test_korean_ocr_rejects_old_recognizer_index(tmp_path):
    from lecture2doc.ocr import OCRConfig, RapidEngine
    from lecture2doc.storage import write

    write(
        tmp_path / "ko.json",
        {
            "models": {"det": "PP-OCRv6-small", "rec": "korean-PP-OCRv5-mobile"},
            "files": {},
        },
    )
    with pytest.raises(RuntimeError, match="ocr-models --language ko"):
        RapidEngine(OCRConfig(language="ko"), directory=tmp_path)



def test_web_api_reruns_existing_korean_ocr_with_current_model(tmp_path, monkeypatch):
    from lecture2doc import ocr

    data = tmp_path / "data"
    job_id = "a" * 24
    folder = data / job_id
    manifest = record(folder)
    previous = recognize_pages(folder, manifest, OCRConfig(language="ko"), engine=Engine())
    manifest["ocr"] = previous
    write(folder / "manifest.json", manifest)

    called = {}

    def rerun(root, current_manifest, config, progress=None, force=False):
        called["root"] = Path(root)
        called["language"] = config.language
        called["force"] = force
        return {
            "status": "complete",
            "language": config.language,
            "version": "test-new-version",
            "elapsed_seconds": 0.25,
            "cache_hits": 0,
            "inference_pages": 1,
            "pages": [{
                "board_page_id": "p1",
                "regions": [],
                "engine": {
                    "models": {"det": "PP-OCRv6-small", "rec": "korean-handwriting-rec"}
                },
            }],
        }

    monkeypatch.setattr(ocr, "recognize_pages", rerun)
    with TestClient(create_app(data)) as client:
        response = client.post(
            f"/api/jobs/{job_id}/ocr",
            headers={"x-lecture2doc": "1"},
        )

    assert response.status_code == 200
    assert called["root"] == folder
    assert called["language"] == "ko"
    assert called["force"] is True
    assert response.json()["recognizer"] == "korean-handwriting-rec"
    updated = read(folder / "manifest.json")
    assert updated["ocr"]["version"] == "test-new-version"
    assert updated["timings"]["ocr"]["status"] == "complete"



def test_korean_recognizer_defaults_to_base_and_handwriting_is_optional(tmp_path, monkeypatch):
    from lecture2doc import ocr

    class FakeFileInfo:
        def __init__(self, *args):
            self.args = args

    class FakeInferSession:
        @staticmethod
        def get_model_url(choice):
            task = choice.args[2]
            if task == "det":
                return {"model_dir": "https://models.example/ppocrv6-det.onnx"}
            return {
                "model_dir": "https://models.example/korean_PP-OCRv5_mobile_rec.onnx",
                "dict_url": "https://models.example/korean_dict.txt",
            }

    rapidocr = types.ModuleType("rapidocr")
    rapidocr.__path__ = []
    rapidocr.inference_engine = types.ModuleType("rapidocr.inference_engine")
    rapidocr.inference_engine.__path__ = []
    base = types.ModuleType("rapidocr.inference_engine.base")
    base.FileInfo = FakeFileInfo
    base.InferSession = FakeInferSession
    utils = types.ModuleType("rapidocr.utils")
    utils.__path__ = []
    typings = types.ModuleType("rapidocr.utils.typings")
    typings.EngineType = types.SimpleNamespace(ONNXRUNTIME="onnxruntime")
    typings.LangDet = types.SimpleNamespace(CH="ch")
    typings.LangRec = types.SimpleNamespace(KOREAN="korean", EN="en")
    typings.ModelType = types.SimpleNamespace(SMALL="small", MOBILE="mobile", MEDIUM="medium")
    typings.OCRVersion = types.SimpleNamespace(PPOCRV5="v5", PPOCRV6="v6")
    typings.TaskType = types.SimpleNamespace(DET="det", REC="rec")
    rapidocr_module = types.ModuleType("rapidocr")
    rapidocr_module.RapidOCR = lambda params: params

    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr_module)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine", rapidocr.inference_engine)
    monkeypatch.setitem(sys.modules, "rapidocr.inference_engine.base", base)
    monkeypatch.setitem(sys.modules, "rapidocr.utils", utils)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", typings)
    monkeypatch.setattr(ocr, "version", lambda _name: "test-version")

    import hashlib
    import json

    def fake_urlopen(url, timeout=60):
        return io.BytesIO(("model-bytes:" + url).encode("utf-8"))

    monkeypatch.setattr(ocr.urllib.request, "urlopen", fake_urlopen)
    ocr.prepare_models("ko", directory=tmp_path)

    monkeypatch.delenv("LECTURE2DOC_KO_RECOGNIZER", raising=False)
    default_engine = ocr.RapidEngine(OCRConfig(language="ko"), directory=tmp_path)
    assert default_engine.identity["models"]["rec"] == "korean-PP-OCRv5-mobile"
    assert default_engine.engine["Rec.model_path"].endswith("korean_PP-OCRv5_mobile_rec.onnx")

    monkeypatch.setenv("LECTURE2DOC_KO_RECOGNIZER", "handwriting")
    handwriting_engine = ocr.RapidEngine(OCRConfig(language="ko"), directory=tmp_path)
    assert handwriting_engine.identity["models"]["rec"] == "korean-handwriting-rec"
    assert handwriting_engine.engine["Rec.model_path"].endswith("rec_korean_hw.onnx")



def test_web_api_stores_qwen_review_without_replacing_original_ocr(tmp_path, monkeypatch):
    from lecture2doc import qwen_vision

    data = tmp_path / "data"
    data.mkdir()
    job_id = "b" * 24
    folder = data / job_id
    manifest = record(folder)
    original_text = "미분방정식 (Dffrential Eqtation)"
    manifest["ocr"] = {
        "status": "complete",
        "language": "ko",
        "pages": [{
            "board_page_id": "p1",
            "input": {"image": "b.png", "sha256": digest(folder / "b.png")},
            "regions": [{"order": 1, "text": original_text, "score": 0.56}],
            "text": original_text,
        }],
    }
    write(folder / "manifest.json", manifest)

    called = {"count": 0}

    def fake_review(image_path, ocr_regions=None, model=None):
        called["count"] += 1
        assert Path(image_path) == folder / "b.png"
        assert ocr_regions[0]["text"] == original_text
        assert model == "qwen3-vl:4b"
        return {
            "status": "complete",
            "version": qwen_vision.QWEN_REVIEW_VERSION,
            "model": "qwen3-vl:4b",
            "source_sha256": digest(image_path),
            "text": "미분방정식 (Differential Equation)",
            "blocks": [{
                "kind": "title",
                "text": "미분방정식 (Differential Equation)",
                "latex": "",
                "confidence": "medium",
                "source_ocr_regions": [1],
            }],
            "local_only": True,
        }

    monkeypatch.setattr(qwen_vision, "selected_model", lambda: "qwen3-vl:4b")
    monkeypatch.setattr(qwen_vision, "review_board_image", fake_review)
    with TestClient(create_app(data)) as client:
        response = client.post(
            f"/api/jobs/{job_id}/qwen-review",
            headers={"x-lecture2doc": "1", "content-type": "application/json"},
            json={"board_page_id": "p1"},
        )
        cached = client.post(
            f"/api/jobs/{job_id}/qwen-review",
            headers={"x-lecture2doc": "1", "content-type": "application/json"},
            json={"board_page_id": "p1"},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "qwen3-vl:4b"
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert called["count"] == 1
    updated = read(folder / "manifest.json")
    page = updated["ocr"]["pages"][0]
    assert page["text"] == original_text
    assert page["qwen_review"]["text"] == "미분방정식 (Differential Equation)"
    assert page["qwen_review"]["local_only"] is True
    assert updated["timings"]["qwen_review"]["board_page_id"] == "p1"
