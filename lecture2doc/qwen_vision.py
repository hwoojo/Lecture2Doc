"""Local Qwen vision and text analysis through the Ollama API."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import cv2

from .storage import digest

# Latest local open-weight Qwen vision-language model as of 2026-11.
LATEST_QWEN_MODEL = "qwen3.8:27b-mlx"
BALANCED_QWEN_MODEL = "qwen3-vl:8b"
COMPACT_QWEN_MODEL = "qwen3-vl:4b"
ULTRA_COMPACT_QWEN_MODEL = "qwen3-vl:2b"
DEFAULT_QWEN_MODEL = LATEST_QWEN_MODEL
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MIN_OLLAMA_VERSION = (0, 12, 7)
MIN_QWEN38_OLLAMA_VERSION = (0, 40, 0)
QWEN_MODEL_CATALOG = {
    LATEST_QWEN_MODEL: {
        "profile": "latest",
        "parameters": "27B",
        "download_gb": 18.0,
        "recommended_memory_gb": 32,
        "platform": "Apple Silicon",
        "vision": True,
        "description": "최신 Qwen3.8-27B MLX 4비트 계열 비전·언어 모델",
    },
    "qwen3.8:27b": {
        "profile": "latest",
        "parameters": "27B",
        "download_gb": 18.0,
        "recommended_memory_gb": 40,
        "platform": "다중 플랫폼",
        "vision": True,
        "description": "최신 Qwen3.8-27B 비전·언어 모델",
    },
    BALANCED_QWEN_MODEL: {
        "profile": "balanced",
        "parameters": "8B",
        "download_gb": 6.2,
        "recommended_memory_gb": 24,
        "platform": "다중 플랫폼",
        "vision": True,
        "description": "메모리가 제한된 장치용 Qwen3-VL 8B",
    },
    COMPACT_QWEN_MODEL: {
        "profile": "compact",
        "parameters": "4B",
        "download_gb": 3.3,
        "recommended_memory_gb": 16,
        "platform": "다중 플랫폼",
        "vision": True,
        "description": "저메모리 Mac용 Qwen3-VL 4B",
    },
    ULTRA_COMPACT_QWEN_MODEL: {
        "profile": "ultra-compact",
        "parameters": "2B",
        "download_gb": 1.9,
        "recommended_memory_gb": 8,
        "platform": "다중 플랫폼",
        "vision": True,
        "description": "메모리가 매우 작은 장치용 Qwen3-VL 2B",
    },
}
QWEN_REVIEW_VERSION = "2026-11-10.qwen-board-review-3-json-robust"
MODEL_SELECTION_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen" / "selection.json"


def ollama_url() -> str:
    return os.environ.get("LECTURE2DOC_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")


def detect_memory_gb() -> float | None:
    """Detect unified/system memory without requiring psutil."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
        if system == "Linux":
            with open("/proc/meminfo", "r", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def recommend_model(profile: str = "auto", memory_gb: float | None = None,
                    system: str | None = None, machine: str | None = None) -> str:
    """Choose the newest practical local vision model for available memory."""
    system = system or platform.system()
    machine = machine or platform.machine().lower()
    memory = detect_memory_gb() if memory_gb is None else memory_gb

    if profile == "latest":
        return LATEST_QWEN_MODEL if system == "Darwin" and machine in ("arm64", "aarch64") else "qwen3.8:27b"
    if profile == "balanced":
        return BALANCED_QWEN_MODEL
    if profile == "compact":
        return COMPACT_QWEN_MODEL
    if profile == "ultra-compact":
        return ULTRA_COMPACT_QWEN_MODEL
    if profile != "auto":
        raise ValueError("Qwen 프로필은 auto, latest, balanced, compact, ultra-compact 중 하나여야 합니다")

    # The 27B 4-bit artifact occupies about 18GB on disk. Keep a margin
    # for macOS, Ollama, the image context, and the rest of Lecture2Doc.
    if memory is not None and memory >= 32:
        if system == "Darwin" and machine in ("arm64", "aarch64"):
            return LATEST_QWEN_MODEL
        if memory >= 40:
            return "qwen3.8:27b"
    if memory is not None and memory >= 24:
        return BALANCED_QWEN_MODEL
    if memory is not None and memory < 16:
        return ULTRA_COMPACT_QWEN_MODEL
    return COMPACT_QWEN_MODEL


def _saved_model() -> str | None:
    try:
        value = json.loads(MODEL_SELECTION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    model = value.get("model") if isinstance(value, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _save_model_selection(model: str) -> None:
    """Persist a successfully installed model choice for the next app launch."""
    MODEL_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_SELECTION_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"model": model, "updated_at": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(MODEL_SELECTION_PATH)


def selected_model() -> str:
    override = os.environ.get("LECTURE2DOC_QWEN_MODEL", "").strip()
    return override or _saved_model() or recommend_model("auto")


def model_download_info(model: str | None = None) -> dict:
    name = model or selected_model()
    details = QWEN_MODEL_CATALOG.get(name, {})
    return {
        "model": name,
        "profile": details.get("profile", "custom"),
        "download_gb": details.get("download_gb"),
        "recommended_memory_gb": details.get("recommended_memory_gb"),
        "system_memory_gb": detect_memory_gb(),
        "description": details.get("description", "사용자가 지정한 Ollama 모델"),
        "vision": details.get("vision", name.startswith("qwen")),
    }


def _version_tuple(version_text: str) -> tuple[int, int, int]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", version_text)[:3])
    return numbers + (0,) * (3 - len(numbers))


def _request_json(path: str, payload: dict | None = None, timeout: int = 15) -> dict:
    url = ollama_url() + path
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API 오류({exc.code}): {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Ollama에 연결할 수 없습니다. Ollama 앱을 설치하고 실행했는지 확인하세요."
        ) from exc


def ollama_status(model: str | None = None) -> dict:
    """Return local Ollama version and whether the requested model is installed."""
    wanted = model or selected_model()
    info = _request_json("/api/version")
    version_text = str(info.get("version", ""))
    minimum = MIN_QWEN38_OLLAMA_VERSION if wanted.startswith("qwen3.8") else MIN_OLLAMA_VERSION
    if _version_tuple(version_text) < minimum:
        required = ".".join(str(part) for part in minimum)
        raise RuntimeError(
            f"Ollama {required} 이상이 필요합니다. "
            f"현재 버전: {version_text or '확인 불가'}"
        )
    tags = _request_json("/api/tags")
    installed = {item.get("name") or item.get("model") for item in tags.get("models", [])}
    return {
        "ollama_version": version_text,
        "model": wanted,
        "installed": wanted in installed,
        "base_url": ollama_url(),
        **model_download_info(wanted),
    }


def pull_model(model: str | None = None, profile: str = "auto") -> dict:
    """Download a recommended or explicitly selected Qwen vision model locally."""
    if model:
        name = model
    elif profile == "auto":
        name = selected_model()
    else:
        name = recommend_model(profile)
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError(
            "Ollama 명령을 찾을 수 없습니다. https://ollama.com/download 에서 macOS용 Ollama를 설치하세요."
        )
    status = ollama_status(name)
    if not status["installed"]:
        completed = subprocess.run([executable, "pull", name], check=False, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"Qwen 모델 다운로드 실패(exit={completed.returncode}). Ollama 앱과 네트워크를 확인하세요."
            )
    status = ollama_status(name)
    if not status["installed"]:
        raise RuntimeError(f"다운로드 명령은 끝났지만 {name} 모델이 설치 목록에 없습니다.")
    _save_model_selection(name)
    return status


def _parse_json_object(content: str, source: str = "Qwen") -> dict:
    """Parse a JSON object even when a model adds a think block or short preamble."""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{source} 응답 내용이 비어 있습니다.")

    # Some reasoning-capable models put reasoning before the final JSON.
    cleaned = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", content, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think\s*>", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        raise RuntimeError(f"{source} 응답에 JSON 본문이 없습니다.")

    candidates = [cleaned]
    fenced = re.search(r"\x60{3}(?:json)?\s*(.*?)\s*\x60{3}", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    # First accept a clean response or a fenced JSON block.
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # Otherwise extract the first balanced object, ignoring braces inside strings.
    for start in (index for index, char in enumerate(cleaned) if char == "{"):
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break

    preview = re.sub(r"\s+", " ", cleaned)[:180]
    raise RuntimeError(f"{source} 응답을 JSON 객체로 읽지 못했습니다. 응답 앞부분: {preview!r}")


def _image_as_base64(image_path: Path, max_side: int = 2200) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Qwen 이미지 읽기 실패: {image_path.name}")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError("Qwen 입력 이미지 인코딩 실패")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def review_board_image(
    image_path: Path,
    ocr_regions: list[dict] | None = None,
    model: str | None = None,
) -> dict:
    """Read a single board image and compare visual evidence with local OCR text."""
    started = time.monotonic()
    image_path = Path(image_path).resolve()
    model_name = model or selected_model()
    status = ollama_status(model_name)
    if not status["installed"]:
        raise RuntimeError(
            f"Qwen 모델 {model_name}이 설치되어 있지 않습니다. "
            "터미널에서 lecture2doc qwen-models를 먼저 실행하세요."
        )

    ocr_hints = [
        {
            "region": int(item.get("order", index + 1)),
            "text": str(item.get("text", ""))[:180],
            "score": round(float(item.get("score", 0)), 3),
        }
        for index, item in enumerate(ocr_regions or [])
        if str(item.get("text", "")).strip()
    ][:80]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "language": {"type": "array", "items": {"type": "string"}},
            "blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["title", "text", "formula", "list"]},
                        "text": {"type": "string"},
                        "latex": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                        "source_ocr_regions": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["kind", "text", "latex", "confidence", "source_ocr_regions"],
                },
            },
            "unreadable_regions": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["language", "blocks", "unreadable_regions", "notes"],
    }
    prompt = (
        "대학교 강의 칠판 이미지를 읽는 역할입니다. 첨부 이미지를 가장 중요한 근거로 사용하세요. "
        "OCR 후보는 참고용이며 틀릴 수 있으니 이미지와 대조해 수정하세요. "
        "한글, 영어, 숫자, 수학 기호와 수식을 원문 그대로 보존하고, 줄과 항목의 읽기 순서를 유지하세요. "
        "수식은 가능한 경우 LaTeX로 작성하고 일반 문장에는 latex를 빈 문자열로 두세요. "
        "읽을 수 없는 부분은 추측하지 말고 unreadable_regions에 위치나 특징을 적으세요. "
        "이미지에 없는 단어·공식·설명을 새로 만들지 마세요. "
        "OCR 후보의 번호는 source_ocr_regions에 기록하되, 확실히 연결할 수 없으면 빈 배열을 쓰세요. "
        "각 블록의 confidence는 시각적으로 판독 가능한 정도이며 실제 정답 확률로 가장하지 마세요. "
        "결과는 지정된 JSON 형식으로만 반환하세요.\n\n"
        "기존 OCR 후보:\n" + json.dumps(ocr_hints, ensure_ascii=False)
    )
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [_image_as_base64(image_path)],
        }],
        "format": schema,
        "stream": False,
        "think": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    response = _request_json("/api/chat", payload, timeout=240)
    message = response.get("message") or {}
    parsed = _parse_json_object(message.get("content", ""), source="Qwen 판서 재검토")
    blocks = parsed.get("blocks")
    if not isinstance(blocks, list):
        raise RuntimeError("Qwen 응답에 blocks 목록이 없습니다.")
    text_lines = [str(item.get("text", "")).strip() for item in blocks if str(item.get("text", "")).strip()]
    return {
        "status": "complete",
        "version": QWEN_REVIEW_VERSION,
        "model": response.get("model") or model_name,
        "source_image": image_path.name,
        "source_sha256": digest(image_path),
        "language": parsed.get("language", []),
        "blocks": blocks,
        "text": "\n".join(text_lines),
        "unreadable_regions": parsed.get("unreadable_regions", []),
        "notes": parsed.get("notes", []),
        "ocr_hints": ocr_hints,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "local_only": True,
    }


def structured_chat(prompt: str, schema: dict, model: str | None = None, timeout: int = 180) -> dict:
    """Run text-only structured generation locally with the same Qwen vision model."""
    model_name = model or selected_model()
    status = ollama_status(model_name)
    if not status["installed"]:
        raise RuntimeError(
            f"Qwen 모델 {model_name}이 설치되어 있지 않습니다. "
            "lecture2doc qwen-models를 먼저 실행하세요."
        )
    response = _request_json(
        "/api/chat",
        {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt + "\n\n반드시 JSON 형식으로만 답변하세요.",
            }],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"temperature": 0, "num_ctx": 8192},
        },
        timeout=timeout,
    )
    message = response.get("message") or {}
    return _parse_json_object(message.get("content", ""), source="Qwen 개념 선별")
