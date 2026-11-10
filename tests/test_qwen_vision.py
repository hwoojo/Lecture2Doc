import base64
import json

import cv2
import numpy as np
import pytest

from lecture2doc import qwen_vision


def test_ollama_status_checks_minimum_version_and_model(monkeypatch):
    def fake_request(path, payload=None, timeout=15):
        if path == "/api/version":
            return {"version": "0.12.7"}
        if path == "/api/tags":
            return {"models": [{"name": "qwen3-vl:4b"}]}
        raise AssertionError(path)

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    status = qwen_vision.ollama_status('qwen3-vl:4b')
    assert status["installed"] is True
    assert status["model"] == "qwen3-vl:4b"
    assert status["ollama_version"] == "0.12.7"


def test_ollama_status_rejects_old_version(monkeypatch):
    monkeypatch.setattr(
        qwen_vision, "_request_json",
        lambda path, payload=None, timeout=15: (
            {"version": "0.12.6"} if path == "/api/version" else {"models": []}
        ),
    )
    with pytest.raises(RuntimeError, match="이상"):
        qwen_vision.ollama_status('qwen3-vl:4b')


def test_qwen_review_sends_board_image_and_preserves_ocr_hints(tmp_path, monkeypatch):
    image_path = tmp_path / "board.png"
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[20:100, 20:220] = (30, 90, 40)
    assert cv2.imwrite(str(image_path), image)

    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": True, "model": model or "qwen3-vl:4b"},
    )
    captured = {}

    def fake_request(path, payload=None, timeout=15):
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout"] = timeout
        return {
            "model": "qwen3-vl:4b",
            "message": {
                "content": json.dumps({
                    "language": ["ko", "en", "math"],
                    "blocks": [{
                        "kind": "formula",
                        "text": "y'' + y' + 2y = 0",
                        "latex": "y'' + y' + 2y = 0",
                        "confidence": "medium",
                        "source_ocr_regions": [1],
                    }],
                    "unreadable_regions": ["오른쪽 아래 작은 글씨"],
                    "notes": [],
                }),
            },
        }

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    result = qwen_vision.review_board_image(
        image_path,
        ocr_regions=[{"order": 1, "text": "y''+y'+2y=0", "score": 0.62}],
        model="qwen3-vl:4b",
    )

    payload = captured["payload"]
    image_data = payload["messages"][0]["images"][0]
    decoded = base64.b64decode(image_data)
    assert captured["path"] == "/api/chat"
    assert captured["timeout"] == 240
    assert payload["model"] == "qwen3-vl:4b"
    assert payload["stream"] is False
    assert "y''+y'+2y=0" in payload["messages"][0]["content"]
    assert decoded.startswith(b"\xff\xd8")
    assert result["status"] == "complete"
    assert result["local_only"] is True
    assert result["blocks"][0]["kind"] == "formula"
    assert result["unreadable_regions"] == ["오른쪽 아래 작은 글씨"]


def test_qwen_review_requires_model_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": False, "model": model or "qwen3-vl:4b"},
    )
    with pytest.raises(RuntimeError, match="qwen-models"):
        qwen_vision.review_board_image(tmp_path / "not-read.png")



def test_recommended_qwen_model_uses_latest_for_large_apple_silicon():
    model = qwen_vision.recommend_model(
        "auto", memory_gb=32, system="Darwin", machine="arm64"
    )
    assert model == "qwen3.8:27b-mlx"


def test_recommended_qwen_model_uses_balanced_and_compact_for_smaller_macs(monkeypatch):
    monkeypatch.setattr(qwen_vision, "detect_memory_gb", lambda: None)
    assert qwen_vision.recommend_model(
        "auto", memory_gb=24, system="Darwin", machine="arm64"
    ) == "qwen3-vl:8b"
    assert qwen_vision.recommend_model(
        "auto", memory_gb=16, system="Darwin", machine="arm64"
    ) == "qwen3-vl:4b"
    assert qwen_vision.recommend_model(
        "auto", memory_gb=8, system="Darwin", machine="arm64"
    ) == "qwen3-vl:2b"
    assert qwen_vision.recommend_model(
        "auto", memory_gb=None, system="Darwin", machine="arm64"
    ) == "qwen3-vl:4b"


def test_recommended_model_profiles_can_be_selected_explicitly():
    assert qwen_vision.recommend_model(
        "latest", memory_gb=8, system="Darwin", machine="arm64"
    ) == "qwen3.8:27b-mlx"
    assert qwen_vision.recommend_model("balanced", memory_gb=16) == "qwen3-vl:8b"
    assert qwen_vision.recommend_model("compact", memory_gb=64) == "qwen3-vl:4b"
    assert qwen_vision.recommend_model("ultra-compact", memory_gb=32) == "qwen3-vl:2b"


def test_selected_model_environment_override(monkeypatch):
    monkeypatch.setenv("LECTURE2DOC_QWEN_MODEL", "qwen3-vl:8b")
    assert qwen_vision.selected_model() == "qwen3-vl:8b"


def test_qwen38_requires_newer_ollama(monkeypatch):
    def fake_request(path, payload=None, timeout=15):
        if path == "/api/version":
            return {"version": "0.39.9"}
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.8:27b-mlx"}]}
        raise AssertionError(path)

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    with pytest.raises(RuntimeError, match="0.40.0"):
        qwen_vision.ollama_status("qwen3.8:27b-mlx")


def test_qwen38_status_accepts_current_ollama_and_reports_model_size(monkeypatch):
    def fake_request(path, payload=None, timeout=15):
        if path == "/api/version":
            return {"version": "0.40.14"}
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.8:27b-mlx"}]}
        raise AssertionError(path)

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    status = qwen_vision.ollama_status("qwen3.8:27b-mlx")
    assert status["installed"] is True
    assert status["download_gb"] == 18.0
    assert status["recommended_memory_gb"] == 32



def test_selected_model_reads_persisted_download_choice(tmp_path, monkeypatch):
    selection = tmp_path / "qwen" / "selection.json"
    monkeypatch.setattr(qwen_vision, "MODEL_SELECTION_PATH", selection)
    monkeypatch.delenv("LECTURE2DOC_QWEN_MODEL", raising=False)
    qwen_vision._save_model_selection("qwen3-vl:8b")
    assert qwen_vision.selected_model() == "qwen3-vl:8b"
    saved = json.loads(selection.read_text(encoding="utf-8"))
    assert saved["model"] == "qwen3-vl:8b"


def test_environment_model_override_has_priority_over_saved_choice(tmp_path, monkeypatch):
    selection = tmp_path / "qwen" / "selection.json"
    monkeypatch.setattr(qwen_vision, "MODEL_SELECTION_PATH", selection)
    monkeypatch.delenv("LECTURE2DOC_QWEN_MODEL", raising=False)
    qwen_vision._save_model_selection("qwen3-vl:8b")
    monkeypatch.setenv("LECTURE2DOC_QWEN_MODEL", "qwen3-vl:4b")
    assert qwen_vision.selected_model() == "qwen3-vl:4b"



def test_qwen_json_parser_removes_thinking_and_markdown_fence():
    content = (
        "<think>먼저 강의 주제를 검토합니다.</think>\n"
        + chr(96) * 3 + "json\n"
        + '{"topic":"미분방정식","concepts":[]}\n'
        + chr(96) * 3
    )
    parsed = qwen_vision._parse_json_object(content)
    assert parsed == {"topic": "미분방정식", "concepts": []}


def test_qwen_json_parser_extracts_object_after_short_preamble():
    content = '분석 결과입니다: {"topic":"선형대수","concepts":[{"name":"고유값"}]} 끝'
    parsed = qwen_vision._parse_json_object(content)
    assert parsed["topic"] == "선형대수"
    assert parsed["concepts"][0]["name"] == "고유값"


def test_structured_chat_disables_thinking_and_uses_robust_json_parser(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": True, "model": model or "qwen3-vl:4b"},
    )

    def fake_request(path, payload=None, timeout=15):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "message": {
                "content": (
                    "<think>핵심 개념 후보를 검토합니다.</think>\n"
                    + chr(96) * 3 + "json\n"
                    + '{"topic":"미분방정식","concepts":[]}\n'
                    + chr(96) * 3
                )
            }
        }

    monkeypatch.setattr(qwen_vision, "_request_json", fake_request)
    parsed = qwen_vision.structured_chat(
        "강의 핵심 개념을 골라 주세요.",
        {"type": "object", "properties": {"topic": {"type": "string"}}},
        model="qwen3-vl:4b",
    )

    assert parsed["topic"] == "미분방정식"
    assert captured["path"] == "/api/chat"
    assert captured["payload"]["think"] is False
    assert captured["payload"]["stream"] is False
