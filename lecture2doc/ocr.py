"""C: local text OCR with native image coordinates and immutable evidence files."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import math
import os
from pathlib import Path
import shutil
import time
import urllib.request

import cv2
import numpy as np

from .storage import digest, key, read, write


KOREAN_HANDWRITING_MODEL_REPO = "Rebeccaaaaaaaa/korean-handwriting-rec"
KOREAN_HANDWRITING_MODEL_FILES = {
    "rec": (
        "https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec/"
        "resolve/main/model/rec_korean_hw.onnx"
    ),
    "dictionary": (
        "https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec/"
        "resolve/main/model/ppocrv5_korean_dict.txt"
    ),
}

OCR_VERSION = "2026-11-10.ocr-3-korean-base-default"


@dataclass(frozen=True)
class OCRConfig:
    language: str = "en"
    min_score: float = 0.5
    review_score: float = 0.75
    preprocessing: str = "rgb"
    threads: int = 4

    def validate(self):
        if self.language not in ("en", "ko"):
            raise ValueError("OCR 언어는 en 또는 ko입니다")
        if not 0 <= self.min_score <= self.review_score <= 1:
            raise ValueError("OCR 점수 범위는 0~1입니다")
        if self.preprocessing not in ("rgb", "adaptive") or self.threads < 1:
            raise ValueError("OCR 전처리 또는 실행 스레드 설정을 확인하세요")
        return self


def models_dir():
    configured = os.environ.get("LECTURE2DOC_OCR_MODELS")
    if configured:
        return Path(configured).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "models" / "ocr"


def prepare_models(language="en", directory=None):
    """Download the detector, default recognizer, and optional Korean handwriting recognizer."""
    OCRConfig(language=language).validate()
    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion, TaskType

    root = Path(directory or models_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)

    detector = FileInfo(
        EngineType.ONNXRUNTIME, OCRVersion.PPOCRV6,
        TaskType.DET, LangDet.CH, ModelType.SMALL,
    )
    files = {}

    def fetch(url, expected=None):
        path = root / Path(url).name
        if not path.exists() or (expected and digest(path) != expected):
            temporary = path.with_name(path.name + ".download")
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as dest:
                    shutil.copyfileobj(response, dest)
                if expected and digest(temporary) != expected:
                    raise ValueError(f"OCR 모델 해시 불일치: {path.name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return dict(path=path.name, sha256=digest(path), url=url)

    detector_info = InferSession.get_model_url(detector)
    files["det"] = fetch(detector_info["model_dir"], detector_info.get("SHA256"))

    if language == "ko":
        # Default remains the official Korean+English PP-OCRv5 model.
        base_config = FileInfo(
            EngineType.ONNXRUNTIME, OCRVersion.PPOCRV5,
            TaskType.REC, LangRec.KOREAN, ModelType.MOBILE,
        )
        base_info = InferSession.get_model_url(base_config)
        files["rec"] = fetch(base_info["model_dir"], base_info.get("SHA256"))
        if base_info.get("dict_url"):
            files["dictionary"] = fetch(base_info["dict_url"])

        # Keep the Korean handwriting fine-tune as an explicit experiment only.
        files["rec_handwriting"] = fetch(KOREAN_HANDWRITING_MODEL_FILES["rec"])
        files["dictionary_handwriting"] = fetch(KOREAN_HANDWRITING_MODEL_FILES["dictionary"])
        recognizer = "korean-PP-OCRv5-mobile"
        recognizer_source = {
            "repository": "PaddleOCR PP-OCRv5 Korean",
            "license": "Apache-2.0",
            "supported_languages": ["Korean", "English"],
            "task": "Korean and English mixed text recognition",
        }
        experimental_source = {
            "repository": KOREAN_HANDWRITING_MODEL_REPO,
            "license": "Apache-2.0",
            "base_model": "PP-OCRv5 Korean Mobile",
            "task": "Korean handwriting and printed-text recognition",
            "selection": "LECTURE2DOC_KO_RECOGNIZER=handwriting",
        }
    else:
        recognizer_config = FileInfo(
            EngineType.ONNXRUNTIME, OCRVersion.PPOCRV6,
            TaskType.REC, LangRec.EN, ModelType.MEDIUM,
        )
        recognizer_info = InferSession.get_model_url(recognizer_config)
        files["rec"] = fetch(recognizer_info["model_dir"], recognizer_info.get("SHA256"))
        if recognizer_info.get("dict_url"):
            files["dictionary"] = fetch(recognizer_info["dict_url"])
        recognizer = "PP-OCRv6-medium"
        recognizer_source = {
            "repository": "RapidOCR model registry",
            "license": "See upstream model metadata",
            "task": "English text recognition",
        }
        experimental_source = None

    result = dict(
        language=language,
        engine="rapidocr",
        engine_version=version("rapidocr"),
        models=dict(det="PP-OCRv6-small", rec=recognizer),
        model_sources=dict(det="RapidOCR PP-OCRv6-small", rec=recognizer_source),
        experimental_models=(
            {"rec": "korean-handwriting-rec", "source": experimental_source}
            if experimental_source else {}
        ),
        files=files,
    )
    write(root / f"{language}.json", result)
    return result


class RapidEngine:
    def __init__(self, config, directory=None):
        root = Path(directory or models_dir()).resolve()
        index = root / f"{config.language}.json"
        if not index.exists():
            raise FileNotFoundError(f"OCR 모델을 준비하세요: lecture2doc ocr-models --language {config.language}")
        self.identity = read(index)
        files = self.identity.get("files", {})

        recognizer_key = "rec"
        dictionary_key = "dictionary"
        selected_recognizer = self.identity.get("models", {}).get("rec", "")
        if config.language == "ko":
            if not {"rec_handwriting", "dictionary_handwriting"} <= files.keys():
                raise RuntimeError(
                    "한국어 OCR 모델 구성이 오래되었습니다. "
                    "'lecture2doc ocr-models --language ko'를 다시 실행하세요."
                )
            mode = os.environ.get("LECTURE2DOC_KO_RECOGNIZER", "base").strip().lower()
            if mode in ("handwriting", "korean-handwriting-rec"):
                recognizer_key = "rec_handwriting"
                dictionary_key = "dictionary_handwriting"
                selected_recognizer = "korean-handwriting-rec"
            elif mode not in ("", "base", "ppocrv5"):
                raise ValueError(
                    "LECTURE2DOC_KO_RECOGNIZER는 base 또는 handwriting이어야 합니다"
                )
            else:
                selected_recognizer = "korean-PP-OCRv5-mobile"

        for item in files.values():
            path = root / item["path"]
            if not path.is_file() or digest(path) != item["sha256"]:
                raise ValueError(f"OCR 모델 파일 확인 실패: {path.name}")

        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec, ModelType, OCRVersion
        selected_model = files.get(recognizer_key)
        if not selected_model:
            raise RuntimeError(
                "선택한 OCR 인식 모델이 준비되지 않았습니다. "
                "'lecture2doc ocr-models --language ko'를 실행하세요."
            )
        params = {
            "Global.use_cls": False, "Global.text_score": config.min_score,
            "Global.max_side_len": 4096, "Global.log_level": "warning",
            "Det.limit_side_len": 1600, "Det.limit_type": "max",
            "Det.model_path": str(root / files["det"]["path"]),
            "Rec.model_path": str(root / selected_model["path"]),
            "Rec.lang_type": LangRec.KOREAN if config.language == "ko" else LangRec.EN,
            "Rec.ocr_version": OCRVersion.PPOCRV5 if config.language == "ko" else OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.MOBILE if config.language == "ko" else ModelType.MEDIUM,
            "EngineConfig.onnxruntime.intra_op_num_threads": config.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
        selected_dictionary = files.get(dictionary_key)
        if selected_dictionary:
            params["Rec.rec_keys_path"] = str(root / selected_dictionary["path"])

        identity_models = dict(self.identity.get("models", {}))
        identity_models["rec"] = selected_recognizer
        self.identity = dict(
            self.identity,
            models=identity_models,
            selected_recognizer=selected_recognizer,
            runtime="onnxruntime",
            runtime_version=version("onnxruntime"),
            installed_engine_version=version("rapidocr"),
        )
        # Paths vary between machines; model bytes and selected recognizer identify the engine.
        self.identity["parameters"] = {
            k: getattr(v, "value", v) for k, v in params.items()
            if not k.endswith(("model_path", "rec_keys_path"))
        }
        self.engine = RapidOCR(params=params)

    def __call__(self, image):
        result = self.engine(image, use_cls=False)
        if result.boxes is None or result.txts is None:
            return []
        return [
            dict(polygon=np.asarray(box).tolist(), text=text, score=float(score))
            for box, text, score in zip(result.boxes, result.txts, result.scores)
        ]


def _path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"OCR 입력 파일 확인 실패: {name}")
    return path


def recognize_pages(root, manifest, config=None, progress=None, force=False, engine=None):
    started = time.monotonic()
    root = Path(root).resolve()
    config = (config or OCRConfig()).validate()
    report = progress or (lambda message: None)
    engine = engine or RapidEngine(config)
    frames = {f["id"]: f for f in manifest["source_frames"]}
    restores = {r["board_page_id"]: r for r in manifest.get("page_reconstructions", [])}
    pages = manifest.get("board_pages", [])
    outputs = []
    cache_hits = 0
    for number, page in enumerate(pages, 1):
        report(f"판서 OCR {number}/{len(pages)}")
        original = frames[page["selected_frame_id"]]
        restored = restores.get(page["id"]) if manifest.get("page_reconstruction_status") == "complete" else None
        image_name = restored["image"] if restored else original["image"]
        source_ids = restored["source_frame_ids"] if restored else [original["id"]]
        provenance = restored.get("provenance") if restored else None
        input_path = _path(root, image_name)
        image_hash = digest(input_path)
        evidence = dict(board_page_id=page["id"], image=image_name, sha256=image_hash,
                        output="B" if restored else "A", source_frame_ids=source_ids,
                        selected_frame_id=original["id"], time=original["time"], provenance=provenance)
        roi = manifest.get("config", {}).get("roi", [0, 0, 1, 1])
        signature = [asdict(config), engine.identity, image_hash, roi, evidence]
        suffix = f"-rerun-{time.time_ns()}" if force else ""
        folder = root / "cache" / "ocr" / (key(signature, OCR_VERSION) + suffix)
        checkpoint = folder / "result.json"
        if checkpoint.exists():
            cached = read(checkpoint)
            # Incomplete caches are rebuilt, rather than advertising missing images.
            if all((root / p).is_file() for p in cached["artifacts"]):
                outputs.append(cached)
                cache_hits += 1
                continue
        started = time.monotonic()
        folder.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"OCR 이미지 읽기 실패: {image_name}")
        height, width = image.shape[:2]
        x, y = int(roi[0]*width), int(roi[1]*height)
        right, bottom = min(width, round((roi[0]+roi[2])*width)), min(height, round((roi[1]+roi[3])*height))
        if not (0 <= x < right <= width and 0 <= y < bottom <= height):
            raise ValueError("OCR 판서 영역을 확인하세요")
        crop = image[y:bottom, x:right].copy()
        if config.preprocessing == "adaptive":
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.cvtColor(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 31, 13), cv2.COLOR_GRAY2BGR)
        raw = engine(crop)
        write(folder / "raw.json", raw)
        regions = []
        overlay = image.copy()
        for item in raw:
            score = float(item["score"])
            points = np.asarray(item["polygon"], dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all() or not math.isfinite(score):
                continue
            if not item["text"].strip() or score < config.min_score:
                continue
            points += [x, y]
            points[:, 0] = np.clip(points[:, 0], x, right-1)
            points[:, 1] = np.clip(points[:, 1], y, bottom-1)
            left, top = np.floor(points.min(axis=0)).astype(int)
            r, b = np.ceil(points.max(axis=0)).astype(int)+1
            if r <= left or b <= top:
                continue
            idx = len(regions)+1
            path = folder / f"region-{idx:04d}.png"
            if not cv2.imwrite(str(path), image[top:b, left:r]):
                raise OSError("OCR 영역 이미지 저장 실패")
            region = dict(id=f"{page['id']}-t{idx:04d}", text=item["text"], score=score,
                          review=score < config.review_score, polygon=points.tolist(),
                          bbox=[int(left), int(top), int(r-left), int(b-top)],
                          image=path.relative_to(root).as_posix(), order=idx)
            regions.append(region)
            color = (40, 140, 240) if region["review"] else (70, 150, 30)
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], True, color, 2)
            cv2.putText(overlay, str(idx), (left, max(15, top-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        def relative(name):
            return (folder / name).relative_to(root).as_posix()
        if not cv2.imwrite(str(folder / "regions.jpg"), overlay):
            raise OSError("OCR 영역 이미지 저장 실패")
        text_value = "\n".join(r["text"] for r in regions)
        (folder / "text.txt").write_text(text_value, encoding="utf-8")
        result = dict(board_page_id=page["id"], status="complete" if regions else "empty",
                      input=evidence, image_size=[width, height], roi=[x, y, right-x, bottom-y],
                      engine=engine.identity, config=asdict(config), algorithm_version=OCR_VERSION,
                      regions=regions, text=text_value, overlay=relative("regions.jpg"),
                      text_file=relative("text.txt"), json_file=relative("result.json"),
                      raw_file=relative("raw.json"), elapsed_seconds=round(time.monotonic()-started, 3))
        result["artifacts"] = [result[k] for k in ("overlay", "text_file", "json_file", "raw_file")] + [r["image"] for r in regions]
        write(checkpoint, result)
        outputs.append(result)
    return dict(status="complete", language=config.language, version=OCR_VERSION, pages=outputs,
                elapsed_seconds=round(time.monotonic()-started, 3), cache_hits=cache_hits,
                inference_pages=len(outputs)-cache_hits)


def annotate_record(record, config=None, output=None, progress=None, force=False):
    started = time.monotonic()
    root = Path(record).resolve()
    manifest = read(root / "manifest.json")
    if output is not None:
        target = Path(output).resolve()
        if target == root or target.is_relative_to(root):
            raise ValueError("OCR 복사본은 기존 기록 밖의 새 폴더에 저장하세요")
        if target.exists():
            raise ValueError("OCR 복사본 출력 폴더가 이미 있습니다")
        def link_or_copy(source, dest):
            # JSON stays independent; large immutable media can share disk blocks.
            if Path(source).suffix == ".json":
                return shutil.copy2(source, dest)
            try:
                return os.link(source, dest)
            except OSError:
                return shutil.copy2(source, dest)
        shutil.copytree(root, target, copy_function=link_or_copy)
        root = target
    result = recognize_pages(root, manifest, config, progress, force)
    result["annotation_elapsed_seconds"] = round(time.monotonic()-started, 3)
    manifest.update(ocr=result, product_version="V1")
    write(root / "manifest.json", manifest)
    return manifest
