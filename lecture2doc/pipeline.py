import copy
import shutil
import time
from pathlib import Path

from . import media
from .reconstruct import reconstruct, scene_context
from .storage import digest, key, stage, write
from .transcript import transcribe
from .vision import Config, analyze
from .pages import build_pages
from .page_restore import restore_pages
from .learning_analysis import build_learning_analysis


def process(video, root, config=None, subtitles=None, model=None, language=None,
            allow_download=False, make_playback=False, progress=None, source_name=None, rerun=(), ocr_language=None):
    root, video = Path(root).resolve(), Path(video).resolve()
    config = (config or Config()).validate()
    root.mkdir(parents=True, exist_ok=True)
    def report(message):
        if progress:
            progress(message)
    started = time.monotonic()
    timings = {}
    def run_stage(stage_root, name, fingerprint, compute, **kwargs):
        begin = time.monotonic()
        computed = False
        def measured(folder):
            nonlocal computed
            computed = True
            return compute(folder)
        try:
            return stage(stage_root, name, fingerprint, measured, **kwargs)
        finally:
            timings[name] = dict(elapsed_seconds=round(time.monotonic()-begin, 3), cache_hit=not computed)
    report("입력 확인")
    fingerprint = digest(video)
    meta = media.probe(video)
    # Own one immutable source copy. Refuse an output directory holding another input.
    from .storage import read
    source_manifest = root / "source.json"
    if source_manifest.exists() and read(source_manifest)["sha256"] != fingerprint:
        raise ValueError("출력 폴더에 다른 영상이 있습니다. 새 폴더를 지정하세요")
    source = root / ("source"+video.suffix.lower())
    if video != source:
        temp = source.with_name(source.name+".tmp")
        shutil.copyfile(video, temp)
        temp.replace(source)
    title = source_name or (read(source_manifest)["name"] if source_manifest.exists() else video.name)
    write(source_manifest, dict(sha256=fingerprint, name=title, path=source.name))
    flags = ["fixed_camera_and_board_roi_require_review", "sampling_can_miss_short_lived_writing", "persistence_can_miss_short_lived_writing"]
    if meta["height"] < 720:
        flags.append("input_below_720p")
    report("원본 해상도 프레임 추출 (캐시 확인)")
    timings["input"] = dict(elapsed_seconds=round(time.monotonic()-started, 3), cache_hit=False)
    visual_error = None
    try:
        frames = run_stage(root, "decode", [fingerprint, config.sample_seconds],
                       lambda folder: media.extract(source, root, folder, config.sample_seconds), force="decode" in rerun)
    except Exception as exc:
        frames = []
        visual_error = f"{type(exc).__name__}: {exc}"
        flags.append("video_decode_failed")
    report("판서 변화 분석")
    analysis_config = config.json()
    analysis_config.pop("reconstruction")
    analysis_config.pop("tile_size")
    try:
        analyzed = run_stage(root, "analysis", [fingerprint, analysis_config],
                         lambda folder: analyze(root, frames, config, meta["duration"]), force="analysis" in rerun)
    except Exception as exc:
        visual_error = f"{type(exc).__name__}: {exc}"
        flags.append("board_analysis_failed_original_frames_retained")
        analyzed = dict(metrics=[], states=[dict(id=f"b{i:05d}", scene=i, start=f["time"],
            end=frames[i+1]["time"] if i+1 < len(frames) else meta["duration"],
            source_frame_ids=[f["id"]], selected_frame_id=f["id"], selection_reason="analysis_failure_original",
            boundary_reason="analysis_failure", quality_flags=["unconfirmed_change"], transcript_ids=[])
            for i, f in enumerate(frames)])
    states = copy.deepcopy(analyzed["states"])
    report("누적 판서 구간과 완성 원본 선정")
    page_analysis = run_stage(root, "pages", [fingerprint, analysis_config],
        lambda folder: build_pages(root, folder, frames, config, meta["duration"], analyzed["metrics"], report),
        force="pages" in rerun)
    pages = copy.deepcopy(page_analysis["pages"])
    report("음성 전사 / 자막 입력")
    transcript_started = time.monotonic()
    try:
        transcript = transcribe(root, source, fingerprint, Path(subtitles) if subtitles else None,
                                model, language, local_only=not allow_download, force="transcript" in rerun)
    except Exception as exc:
        transcript = dict(status="failed", origin="subtitles" if subtitles else "faster-whisper",
                          segments=[], message=f"{type(exc).__name__}: {exc}")
        flags.append("transcript_failed")
    timings["transcript"] = dict(elapsed_seconds=round(time.monotonic()-transcript_started, 3), status=transcript["status"])
    # Every transcript remains globally accessible, including gaps and out-of-video cues.
    for s in states + pages:
        s["transcript_ids"] = [t["id"] for t in transcript["segments"] if t["end"] > s["start"] and t["start"] < s["end"]]
    by_id = {f["id"]: f for f in frames}
    report("판서 복원")
    def restore(folder):
        results = []
        contexts = scene_context(root, folder, states, by_id, config) if config.reconstruction else {}
        for number, s in enumerate(states, 1):
            checkpoint = folder / (s["id"]+".result.json")
            if checkpoint.exists():
                results.append(read(checkpoint))
                if number % 25 == 0 or number == len(states):
                    report(f"판서 복원 {number}/{len(states)} (캐시 포함)")
                continue
            try:
                results.append(reconstruct(root, folder, s, by_id, config, context=contexts.get(s["scene"])))
            except Exception as exc:
                results.append(dict(id="r"+s["id"][1:], board_state_id=s["id"], status="fallback",
                                    image=by_id[s["selected_frame_id"]]["image"],
                                    source_frame_ids=[s["selected_frame_id"]], provenance=None,
                                    quality_flags=["reconstruction_failed", str(exc)], coverage=None))
            write(checkpoint, results[-1])
            if number % 25 == 0 or number == len(states):
                report(f"판서 복원 {number}/{len(states)}")
        return results
    reconstructions = run_stage(root, "reconstruction", [fingerprint, config.json(), analyzed["states"]], restore,
                            force="reconstruction" in rerun)
    report("누적 판서 B 복원")
    page_reconstructions=run_stage(root,"page-restoration",[fingerprint,config.json(),pages,page_analysis.get('policy'),analyzed['states']],
        lambda folder: restore_pages(root,folder,pages,frames,config,page_analysis.get('policy'),states,report),
        force='page-restoration' in rerun)
    playback_path = source.name
    if make_playback or not media.browser_compatible(meta, source.suffix):
        report("브라우저 재생용 영상 변환")
        def convert(folder):
            dest = folder / "playback.mp4"
            media.playback(source, dest)
            return dict(path=dest.relative_to(root).as_posix())
        try:
            playback_path = run_stage(root, "playback", fingerprint, convert, force="playback" in rerun)["path"]
        except Exception:
            flags.append("playback_conversion_failed_original_available")
    result = dict(schema_version=1, source=dict(name=title, path=source.name, sha256=fingerprint, **meta),
                  playback=playback_path, config=config.json(), source_frames=frames,
                  primary_output="board_pages", page_reconstruction_status="complete" if config.reconstruction else "disabled",
                  page_reconstructions=page_reconstructions,
                  board_pages=pages, page_policy=page_analysis.get("policy"), page_events=page_analysis["events"],
                  board_states=states, selected_frames=[dict(board_state_id=s["id"],
                     source_frame_id=s["selected_frame_id"], reason=s["selection_reason"],
                     alternatives=s.get("selection_candidates", [])) for s in states],
                  reconstructed_boards=reconstructions, transcript=transcript,
                  quality_flags=flags, analysis_metrics=analyzed["metrics"],
                  elapsed_seconds=round(time.monotonic()-started, 3),
                  future_annotations=[], temporal_links_only=True, visual_error=visual_error)
    write(root / "manifest.json", result)
    if ocr_language:
        from .ocr import OCRConfig, recognize_pages
        report("판서 텍스트 인식")
        ocr_started = time.monotonic()
        try:
            result["ocr"] = recognize_pages(root, result, OCRConfig(language=ocr_language), report, force="ocr" in rerun)
        except Exception as exc:
            result["ocr"] = dict(status="failed", language=ocr_language, pages=[], message=f"{type(exc).__name__}: {exc}")
        timings["ocr"] = dict(elapsed_seconds=round(time.monotonic()-ocr_started, 3),
                              status=result["ocr"]["status"], cache_hits=result["ocr"].get("cache_hits", 0))
    report("강의 중요도와 개념 등장 시간 분석")
    learning_started = time.monotonic()
    try:
        result["learning_analysis"] = build_learning_analysis(
            root, result, force="learning-analysis" in rerun
        )
    except Exception as exc:
        result["learning_analysis"] = dict(
            status="failed", method="evidence_rules", pages=[], concepts=[],
            message=f"{type(exc).__name__}: {exc}",
        )
    timings["learning_analysis"] = dict(
        elapsed_seconds=round(time.monotonic()-learning_started, 3),
        status=result["learning_analysis"]["status"],
    )
    result.update(product_version="V1", timings=timings, elapsed_seconds=round(time.monotonic()-started, 3),
                  elapsed_scope="video_input_through_learning_analysis_excluding_final_manifest_write")
    write(root / "manifest.json", result)
    report("완료")
    return result
