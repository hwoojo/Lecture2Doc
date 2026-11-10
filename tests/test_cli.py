import sys
import types

from lecture2doc import cli


def test_setup_models_downloads_both_ocr_languages_and_auto_qwen(monkeypatch, capsys):
    calls = []
    ocr = types.ModuleType("lecture2doc.ocr")

    def prepare_models(language):
        calls.append(("ocr", language))
        return {"models": {"det": "PP-OCRv6-small", "rec": f"{language}-recognizer"}}

    ocr.prepare_models = prepare_models

    qwen = types.ModuleType("lecture2doc.qwen_vision")
    qwen.recommend_model = lambda profile: "qwen3.8:27b-mlx"
    qwen.ollama_status = lambda model: calls.append(("check", model)) or {
        "installed": False, "model": model, "ollama_version": "0.40.14"
    }
    qwen.pull_model = lambda model: calls.append(("pull", model)) or {
        "model": model, "download_gb": 18.0, "ollama_version": "0.40.14"
    }

    monkeypatch.setitem(sys.modules, "lecture2doc.ocr", ocr)
    monkeypatch.setitem(sys.modules, "lecture2doc.qwen_vision", qwen)
    monkeypatch.setattr(sys, "argv", ["lecture2doc", "setup-models"])

    cli.main()

    assert calls == [
        ("check", "qwen3.8:27b-mlx"),
        ("ocr", "en"),
        ("ocr", "ko"),
        ("pull", "qwen3.8:27b-mlx"),
    ]
    assert "필수 로컬 모델 세트 준비 완료" in capsys.readouterr().out


def test_setup_models_honors_selected_languages_and_qwen_profile(monkeypatch):
    calls = []
    ocr = types.ModuleType("lecture2doc.ocr")
    ocr.prepare_models = lambda language: calls.append(("ocr", language)) or {"models": {}}

    qwen = types.ModuleType("lecture2doc.qwen_vision")
    qwen.selected_model = lambda: "automatic"
    qwen.recommend_model = lambda profile: "qwen3-vl:8b"
    qwen.ollama_status = lambda model: calls.append(("check", model)) or {"installed": False}
    qwen.pull_model = lambda model: calls.append(("pull", model)) or {
        "model": model, "download_gb": 6.2, "ollama_version": "0.40.14"
    }

    monkeypatch.setitem(sys.modules, "lecture2doc.ocr", ocr)
    monkeypatch.setitem(sys.modules, "lecture2doc.qwen_vision", qwen)
    monkeypatch.setattr(
        sys, "argv",
        ["lecture2doc", "setup-models", "--qwen-profile", "balanced", "--ocr-language", "ko"],
    )

    cli.main()

    assert calls == [
        ("check", "qwen3-vl:8b"),
        ("ocr", "ko"),
        ("pull", "qwen3-vl:8b"),
    ]


def test_setup_models_checks_ollama_before_downloading_ocr(monkeypatch):
    ocr = types.ModuleType("lecture2doc.ocr")
    calls = []
    ocr.prepare_models = lambda language: calls.append(("ocr", language))

    qwen = types.ModuleType("lecture2doc.qwen_vision")
    qwen.selected_model = lambda: "qwen3-vl:4b"
    qwen.recommend_model = lambda profile: "qwen3-vl:4b"

    def fail_status(model):
        raise RuntimeError("Ollama 연결 실패")

    qwen.ollama_status = fail_status
    qwen.pull_model = lambda model: calls.append(("pull", model))

    monkeypatch.setitem(sys.modules, "lecture2doc.ocr", ocr)
    monkeypatch.setitem(sys.modules, "lecture2doc.qwen_vision", qwen)
    monkeypatch.setattr(sys, "argv", ["lecture2doc", "setup-models"])

    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Ollama 오류가 setup-models에서 처리되어야 합니다")

    assert calls == []
