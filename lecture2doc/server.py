"""Single-user loopback server. Background processing never blocks media requests."""
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import re
import secrets
import shutil
import time
from threading import RLock
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .pipeline import process
from .storage import digest, key, read, write
from .vision import Config


def code_revision():
    package = Path(__file__).parent
    value = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        if path.is_file():
            value.update(path.relative_to(package).as_posix().encode())
            value.update(path.read_bytes())
    return value.hexdigest()


class BoardSelection(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=10000)


class PurgeConfirmation(BaseModel):
    confirmed: bool = False


class QwenReviewRequest(BaseModel):
    board_page_id: str = Field(min_length=1, max_length=128)


def create_app(data_dir=None, record_dir=None):
    project_root = Path(__file__).resolve().parent.parent
    root = Path(data_dir).expanduser().resolve() if data_dir is not None else project_root / "data"
    root.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lecture2doc")
    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True)
    app = FastAPI(title="Lecture2Doc", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
    app.state.pool = pool
    app.state.futures = {}
    app.state.run_clocks = {}
    completed_times = {}
    revision = code_revision()
    imported = Path(record_dir).resolve() if record_dir else None
    imported_id = "000000000000000000000001"
    edit_lock = RLock()
    library_file = root / "library-edits.json"

    def library_edits():
        edits = read(library_file) if library_file.exists() else {"deleted": {}}
        edits.setdefault("purged_imports", [])
        return edits

    def library_key(job_id):
        return "import:" + str(imported) if imported and job_id == imported_id else job_id
    if imported and not (imported / "manifest.json").is_file():
        raise ValueError("--record 폴더에 manifest.json이 없습니다")

    @app.middleware("http")
    async def local_requests(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if request.headers.get("x-lecture2doc") != "1" or (origin and urlsplit(origin).netloc != request.headers.get("host")):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "동일 출처 앱 요청만 허용됩니다"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        if request.url.path == "/" or request.url.path.startswith("/static/") or (request.url.path.startswith("/api/") and "/files/" not in request.url.path):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/app")
    def app_info():
        return dict(app="lecture2doc", project=str(Path(__file__).resolve().parent.parent),
                    data=str(root), record=str(imported) if imported else None, code_revision=revision)

    def job_dir(job_id, include_deleted=False):
        if not re.fullmatch(r"[a-f0-9]{24}", job_id):
            raise HTTPException(404)
        if library_key(job_id) in library_edits()["purged_imports"]:
            raise HTTPException(404)
        if not include_deleted and library_key(job_id) in library_edits()["deleted"]:
            raise HTTPException(404, "삭제한 강의입니다")
        if imported and job_id == imported_id:
            return imported
        d = root / job_id
        if not d.is_dir():
            raise HTTPException(404)
        return d

    def active(job_id):
        future = app.state.futures.get(job_id)
        return future is not None and not future.done()

    def current_status(item, job_id):
        item = dict(item)
        if item["status"] in ("running", "queued") and not active(job_id):
            item.update(status="interrupted", message="서버가 중단되었습니다. CLI로 같은 폴더를 다시 처리하세요")
        clock = app.state.run_clocks.get(job_id)
        if item["status"] == "running" and clock is not None:
            item["elapsed_seconds"] = round(time.monotonic()-clock, 3)
        if item["status"] == "queued" and item.get("queued_at"):
            item["elapsed_seconds"] = round(max(0, time.time()-item["queued_at"]), 3)
        if item["status"] == "complete" and "elapsed_seconds" not in item:
            # Older workers still finish normally during a graceful reload.
            path = root / job_id / "manifest.json"
            if path.is_file():
                stamp = path.stat().st_mtime_ns
                if completed_times.get(job_id, (None,))[0] != stamp:
                    result = read(path)
                    elapsed = result.get("elapsed_seconds") if result.get("elapsed_scope") else None
                    completed_times[job_id] = (stamp, elapsed)
                elapsed = completed_times[job_id][1]
                if elapsed is not None:
                    item["elapsed_seconds"] = elapsed
        return item

    def board_key(m, field, board):
        return key([m["source"].get("sha256"), field, board["id"], board.get("start"),
                    board.get("end"), board.get("selected_frame_id")], "user-board-edits-1")

    def board_edits(folder):
        path = folder / "user-edits.json"
        return read(path) if path.exists() else {"deleted_boards": {}}

    def visible_manifest(folder):
        m = read(folder / "manifest.json")
        field = "board_pages" if "board_pages" in m else "board_states"
        deleted = board_edits(folder)["deleted_boards"]
        visible, removed = [], []
        for number, board in enumerate(m[field], 1):
            board["display_number"] = number
            if board_key(m, field, board) in deleted:
                removed.append(dict(id=board["id"], display_number=number, start=board["start"], end=board["end"]))
            else:
                visible.append(board)
        m[field] = visible
        m["deleted_boards"] = removed
        ids = {board["id"] for board in visible}
        if field == "board_pages":
            m["page_reconstructions"] = [r for r in m.get("page_reconstructions", []) if r["board_page_id"] in ids]
            if "ocr" in m:
                m["ocr"]["pages"] = [p for p in m["ocr"].get("pages", []) if p["board_page_id"] in ids]
        else:
            m["reconstructed_boards"] = [r for r in m.get("reconstructed_boards", []) if r["board_state_id"] in ids]
        return m

    def run(job_id, video, config, subs, model, language, download, playback, title, ocr_language):
        folder = root / job_id
        started = time.time()
        app.state.run_clocks[job_id] = time.monotonic()
        queued = read(folder / "status.json").get("queued_at", started)
        def update(message, status="running"):
            value = dict(id=job_id, status=status, message=message, queued_at=queued,
                         started_at=started, updated_at=time.time(),
                         elapsed_seconds=round(time.monotonic()-app.state.run_clocks[job_id], 3))
            if status != "running":
                value["finished_at"] = value["updated_at"]
            write(folder / "status.json", value)
        try:
            update("처리 시작")
            result = process(video, folder, config, subs, model or None, language or None,
                    download, playback, progress=update, source_name=title, ocr_language=ocr_language or None)
            message = "A/B 완료 · OCR 처리 실패" if result.get("ocr", {}).get("status") == "failed" else "완료"
            update(message, "complete")
        except Exception as exc:
            update(f"{type(exc).__name__}: {exc}", "failed")

    @app.get("/api/jobs")
    def jobs():
        items = []
        edits = library_edits()
        deleted = edits["deleted"]
        if imported and library_key(imported_id) not in deleted and library_key(imported_id) not in edits["purged_imports"]:
            items.append(dict(id=imported_id, status="complete", message="CLI 분석 결과",
                              name=read(imported / "manifest.json")["source"]["name"], can_delete=True))
        for p in sorted(root.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not re.fullmatch(r"[a-f0-9]{24}", p.parent.name) or library_key(p.parent.name) in deleted:
                continue
            if imported and p.parent.resolve() == imported:
                continue
            item = current_status(read(p), p.parent.name)
            if (p.parent / "source.json").exists():
                item["name"] = read(p.parent / "source.json")["name"]
            item["can_delete"] = not active(p.parent.name)
            items.append(item)
        return items

    @app.get("/api/trash")
    def trash():
        items = []
        for identity, item in library_edits()["deleted"].items():
            if identity.startswith("import:") and identity != library_key(imported_id):
                continue
            items.append(dict(item, purge_scope="library_entry" if identity.startswith("import:") else "files"))
        return sorted(items, key=lambda item: item["deleted_at"], reverse=True)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        with edit_lock:
            folder = job_dir(job_id, include_deleted=True)
            if active(job_id):
                raise HTTPException(409, "처리 중인 강의는 완료 후 삭제할 수 있습니다")
            m = read(folder / "manifest.json") if (folder / "manifest.json").exists() else None
            source = read(folder / "source.json") if (folder / "source.json").exists() else {}
            edits = library_edits()
            edits["deleted"].setdefault(library_key(job_id), dict(id=job_id,
                name=m["source"]["name"] if m else source.get("name", "강의"),
                deleted_at=datetime.now(timezone.utc).isoformat()))
            write(library_file, edits)
        return dict(status="deleted")

    @app.post("/api/jobs/{job_id}/restore")
    def restore_job(job_id: str):
        with edit_lock:
            job_dir(job_id, include_deleted=True)
            edits = library_edits()
            edits["deleted"].pop(library_key(job_id), None)
            write(library_file, edits)
        return dict(status="restored")

    @app.post("/api/jobs/{job_id}/purge")
    def purge_job(job_id: str, confirmation: PurgeConfirmation):
        if not confirmation.confirmed:
            raise HTTPException(422, "영구 삭제 확인이 필요합니다")
        if not re.fullmatch(r"[a-f0-9]{24}", job_id):
            raise HTTPException(404)
        with edit_lock:
            edits = library_edits()
            identity = library_key(job_id)
            if identity not in edits["deleted"]:
                raise HTTPException(409, "휴지통에 있는 강의만 영구 삭제할 수 있습니다")
            if active(job_id):
                raise HTTPException(409, "처리 중인 강의는 완료 후 삭제할 수 있습니다")
            if imported and job_id == imported_id:
                # --record is an external reference, not an owned upload folder.
                if identity not in edits["purged_imports"]:
                    edits["purged_imports"].append(identity)
            else:
                folder = root / job_id
                if folder.is_symlink() or folder.resolve().parent != root:
                    raise HTTPException(403, "강의 저장 폴더를 확인할 수 없습니다")
                try:
                    if folder.exists():
                        shutil.rmtree(folder)
                except OSError:
                    raise HTTPException(500, "일부 파일을 삭제하지 못했습니다. 영구 삭제를 다시 시도하세요")
                app.state.futures.pop(job_id, None)
                app.state.run_clocks.pop(job_id, None)
            edits["deleted"].pop(identity)
            write(library_file, edits)
        return dict(status="purged")

    def edit_boards(job_id, selection, deleting):
        with edit_lock:
            folder = job_dir(job_id)
            if active(job_id):
                raise HTTPException(409, "처리 중인 강의는 완료 후 편집할 수 있습니다")
            if not (folder / "manifest.json").exists():
                raise HTTPException(404)
            m = read(folder / "manifest.json")
            field = "board_pages" if "board_pages" in m else "board_states"
            boards = {b["id"]: b for b in m[field]}
            if not set(selection.ids) <= boards.keys():
                raise HTTPException(404, "판서를 찾을 수 없습니다")
            edits = board_edits(folder)
            for identity in set(selection.ids):
                signature = board_key(m, field, boards[identity])
                if deleting:
                    edits["deleted_boards"][signature] = dict(id=identity, deleted_at=datetime.now(timezone.utc).isoformat())
                else:
                    edits["deleted_boards"].pop(signature, None)
            write(folder / "user-edits.json", edits)
        return dict(status="deleted" if deleting else "restored", ids=selection.ids)

    @app.post("/api/jobs/{job_id}/boards/delete")
    def delete_boards(job_id: str, selection: BoardSelection):
        return edit_boards(job_id, selection, True)

    @app.post("/api/jobs/{job_id}/boards/restore")
    def restore_boards(job_id: str, selection: BoardSelection):
        return edit_boards(job_id, selection, False)

    @app.post("/api/jobs", status_code=202)
    async def upload(video: UploadFile = File(...), subtitles: UploadFile | None = File(None),
                     sample_seconds: float = Form(.5), roi: str = Form("0,0,1,1"),
                     board: str = Form("auto"), model: str = Form(""), language: str = Form(""),
                     reconstruction: bool = Form(True), allow_download: bool = Form(False), make_playback: bool = Form(False),
                     ocr_language: str = Form("")):
        if ocr_language not in ("", "en", "ko"):
            raise HTTPException(422, "OCR 언어를 확인하세요")
        try:
            config = Config(reconstruction=reconstruction, sample_seconds=sample_seconds, board=board, roi=tuple(float(v) for v in roi.split(","))).validate()
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        ext = Path(video.filename or "").suffix.lower()
        if ext not in (".mp4", ".mkv", ".mov", ".webm", ".avi", ".mts", ".m4v"):
            raise HTTPException(422, "지원 영상 확장자가 아닙니다")
        subext = Path(subtitles.filename or "").suffix.lower() if subtitles else None
        if subext and subext not in (".srt", ".vtt", ".json"):
            raise HTTPException(422, "자막은 SRT/VTT/JSON을 지원합니다")
        job_id = secrets.token_hex(12)
        folder = root / job_id
        folder.mkdir()
        async def save(upload, path, limit):
            size = 0
            with path.open("wb") as f:
                while chunk := await upload.read(1024*1024):
                    size += len(chunk)
                    if size > limit:
                        path.unlink(missing_ok=True)
                        raise HTTPException(413, "파일 크기 제한 초과")
                    f.write(chunk)
            await upload.close()
        path = folder / ("source"+ext)
        await save(video, path, 20*1024**3)
        subpath = folder / ("subtitles"+subext) if subtitles else None
        if subtitles:
            await save(subtitles, subpath, 20*1024**2)
        write(folder / "status.json", dict(id=job_id, status="queued", message="처리 대기", queued_at=time.time(), elapsed_seconds=0))
        app.state.futures[job_id] = pool.submit(run, job_id, path, config, subpath, model, language,
                                                allow_download, make_playback, video.filename, ocr_language)
        return dict(id=job_id)

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        p = job_dir(job_id) / "status.json"
        return current_status(read(p), job_id) if p.exists() else dict(status="unknown")

    @app.get("/api/jobs/{job_id}/manifest")
    def manifest(job_id: str):
        p = job_dir(job_id) / "manifest.json"
        if not p.exists():
            raise HTTPException(404, "분석 결과가 아직 없습니다")
        return visible_manifest(p.parent)

    @app.post("/api/jobs/{job_id}/ocr")
    def rerun_ocr(job_id: str):
        with edit_lock:
            folder = job_dir(job_id)
            if active(job_id):
                raise HTTPException(409, "강의 처리 중에는 OCR을 실행할 수 없습니다")
            path = folder / "manifest.json"
            if not path.is_file():
                raise HTTPException(404, "분석 결과가 아직 없습니다")
            manifest_data = read(path)
            previous = manifest_data.get("ocr") or {}
            language = previous.get("language") or manifest_data.get("config", {}).get("ocr_language") or "ko"
            previous_pages = previous.get("pages") or []
            previous_config = next(
                (page.get("config") for page in previous_pages if page.get("config")),
                {},
            )
            try:
                from .ocr import OCRConfig, recognize_pages
                config = OCRConfig(
                    language=language,
                    min_score=float(previous_config.get("min_score", 0.5)),
                    review_score=float(previous_config.get("review_score", 0.75)),
                    preprocessing=previous_config.get("preprocessing", "rgb"),
                    threads=int(previous_config.get("threads", 4)),
                ).validate()
                started = time.monotonic()
                result = recognize_pages(
                    folder, manifest_data, config=config, progress=lambda message: None, force=True
                )
                manifest_data["ocr"] = result
                manifest_data.setdefault("timings", {})["ocr"] = dict(
                    elapsed_seconds=result.get("elapsed_seconds", round(time.monotonic() - started, 3)),
                    status=result.get("status"),
                    cache_hits=result.get("cache_hits", 0),
                    inference_pages=result.get("inference_pages", 0),
                )
                write(path, manifest_data)
            except Exception as exc:
                raise HTTPException(500, f"OCR 재실행 실패: {type(exc).__name__}: {exc}")
        first_page = next(iter(result.get("pages", [])), {})
        recognizer = (first_page.get("engine") or {}).get("models", {}).get("rec")
        return dict(
            status="complete",
            language=result.get("language"),
            pages=len(result.get("pages", [])),
            recognizer=recognizer,
            elapsed_seconds=result.get("elapsed_seconds"),
            message=f"OCR 재실행 완료 · {recognizer or '모델 정보 없음'}",
        )

    @app.post("/api/jobs/{job_id}/qwen-review")
    def run_qwen_review(job_id: str, request: QwenReviewRequest):
        with edit_lock:
            folder = job_dir(job_id)
            if active(job_id):
                raise HTTPException(409, "강의 처리 중에는 Qwen 판서 재검토를 실행할 수 없습니다")
            manifest_path = folder / "manifest.json"
            if not manifest_path.is_file():
                raise HTTPException(404, "분석 결과가 아직 없습니다")
            manifest_data = read(manifest_path)
            ocr_data = manifest_data.get("ocr") or {}
            ocr_page = next(
                (page for page in ocr_data.get("pages", [])
                 if page.get("board_page_id") == request.board_page_id),
                None,
            )
            if ocr_page is None:
                raise HTTPException(404, "선택한 판서의 OCR 결과가 없습니다")
            image_name = (ocr_page.get("input") or {}).get("image")
            if not image_name:
                raise HTTPException(404, "판서 원본 이미지 경로가 없습니다")
            image_path = (folder / image_name).resolve()
            if not image_path.is_relative_to(folder.resolve()) or not image_path.is_file():
                raise HTTPException(404, "판서 이미지 파일을 찾을 수 없습니다")
            try:
                from .qwen_vision import QWEN_REVIEW_VERSION, review_board_image, selected_model
                model_name = selected_model()
                existing = ocr_page.get("qwen_review") or {}
                if (
                    existing.get("status") == "complete"
                    and existing.get("version") == QWEN_REVIEW_VERSION
                    and existing.get("model") == model_name
                    and existing.get("source_sha256") == digest(image_path)
                ):
                    return dict(
                        status="complete",
                        board_page_id=request.board_page_id,
                        model=model_name,
                        blocks=len(existing.get("blocks", [])),
                        elapsed_seconds=0,
                        cached=True,
                        message=f"기존 Qwen 결과 재사용 · {model_name}",
                    )
                started = time.monotonic()
                result = review_board_image(
                    image_path,
                    ocr_regions=ocr_page.get("regions", []),
                    model=model_name,
                )
            except Exception as exc:
                raise HTTPException(500, f"Qwen 판서 재검토 실패: {type(exc).__name__}: {exc}")
            ocr_page["qwen_review"] = result
            manifest_data.setdefault("timings", {})["qwen_review"] = dict(
                elapsed_seconds=round(time.monotonic() - started, 3),
                status=result.get("status"),
                model=result.get("model"),
                board_page_id=request.board_page_id,
            )
            write(manifest_path, manifest_data)
        return dict(
            status="complete",
            board_page_id=request.board_page_id,
            model=result.get("model"),
            blocks=len(result.get("blocks", [])),
            elapsed_seconds=result.get("elapsed_seconds"),
            message=f"Qwen 판서 재검토 완료 · {result.get('model')}",
        )

    @app.post("/api/jobs/{job_id}/learning-analysis")
    def run_learning_analysis(job_id: str):
        with edit_lock:
            folder = job_dir(job_id)
            if active(job_id):
                raise HTTPException(409, "강의 처리 중에는 학습 분석을 실행할 수 없습니다")
            path = folder / "manifest.json"
            if not path.is_file():
                raise HTTPException(404, "분석 결과가 아직 없습니다")
            manifest_data = read(path)
            started = time.monotonic()
            try:
                from .learning_analysis import build_learning_analysis
                analysis = build_learning_analysis(folder, manifest_data, force=True)
            except Exception as exc:
                raise HTTPException(500, f"학습 분석 실패: {type(exc).__name__}: {exc}")
            manifest_data["learning_analysis"] = analysis
            manifest_data.setdefault("timings", {})["learning_analysis"] = dict(
                elapsed_seconds=round(time.monotonic() - started, 3),
                status=analysis["status"],
            )
            write(path, manifest_data)
        return dict(status="complete", analysis_status=analysis["status"], message=analysis.get("message", ""))

    @app.get("/api/jobs/{job_id}/files/{name:path}")
    def files(job_id: str, name: str):
        folder = job_dir(job_id)
        if not (folder / "manifest.json").exists():
            raise HTTPException(404)
        m = read(folder / "manifest.json")
        allowed = {m["source"]["path"], m["playback"]}
        allowed.update(f["image"] for f in m["source_frames"])
        restores=m["reconstructed_boards"]+m.get("page_reconstructions",[])
        allowed.update(path for r in restores for path in r.get("diagnostics",[]))
        allowed.update(r["image"] for r in restores)
        allowed.update(r["provenance"] for r in restores if r["provenance"])
        for page in m.get("ocr", {}).get("pages", []):
            allowed.update(page.get("artifacts", []))
        p = (folder / name).resolve()
        if name not in allowed or not p.is_relative_to(folder) or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        html = (static / "index.html").read_text(encoding="utf-8")
        # A new Cache-Control header cannot invalidate an already-fresh browser
        # entry. Content-based URLs also replace scripts cached by older builds.
        for name in ("app.js", "style.css"):
            revision = hashlib.sha256((static / name).read_bytes()).hexdigest()[:16]
            html = html.replace(f"/static/{name}", f"/static/{name}?v={revision}")
        return HTMLResponse(html)

    return app
