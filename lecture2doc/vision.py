"""Model-free fixed-camera board measurements. No generated/inpainted pixels."""
from dataclasses import asdict, dataclass
from functools import lru_cache

import cv2
import numpy as np

from .feature_cache import (
    FEATURE_CHUNK_SIZE,
    feature_cache_dir,
    write_feature_chunk,
)


@dataclass
class Config:
    sample_seconds: float = 0.5
    analysis_width: int = 640
    board: str = "auto"
    roi: tuple = (0.0, 0.0, 1.0, 1.0)
    confirm: int = 2
    change_pixels: int = 35
    change_fraction: float = 0.001
    cut_fraction: float = 0.65
    tile_size: int = 32
    reconstruction: bool = True

    def validate(self):
        if not 0.05 <= self.sample_seconds <= 10:
            raise ValueError("sample_seconds 범위는 0.05~10초입니다")
        if not 160 <= self.analysis_width <= 1280 or not 1 <= self.confirm <= 10:
            raise ValueError("분석 해상도 또는 확인 횟수가 범위를 벗어났습니다")
        if self.board not in ("auto", "blackboard", "whiteboard"):
            raise ValueError("board는 auto/blackboard/whiteboard입니다")
        x, y, w, h = self.roi
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1 or y + h > 1:
            raise ValueError("ROI는 정규화된 x,y,w,h이며 영상 안에 있어야 합니다")
        return self

    def json(self):
        return asdict(self)


def crop(image, roi):
    h, w = image.shape[:2]
    x, y, rw, rh = roi
    return image[int(y*h):max(int((y+rh)*h), int(y*h)+1),
                 int(x*w):max(int((x+rw)*w), int(x*w)+1)]


def small(image, config):
    a = crop(image, config.roi)
    width = min(config.analysis_width, a.shape[1])
    return cv2.resize(a, (width, max(1, round(a.shape[0]*width/a.shape[1]))), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=8)
def _surface_basis(h, w):
    yy, xx = np.mgrid[:h, :w].astype("float32")
    xx, yy = 2*xx/w-1, 2*yy/h-1
    return np.stack((xx*0+1, xx, yy, xx*yy, xx*xx, yy*yy), axis=-1)


def surface_foreground(image, polarity=None, return_core=False):
    """Fit a smooth illumination field, not a single board color.

    Robust coarse-cell regression excludes large departures. A color departure
    must contain a thick core: a stroke touching a person cannot flood-fill the
    entire connected writing component into the foreground mask.
    The fitted image is analysis-only and is never used as reconstructed pixels.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if polarity is None:
        detail = gray.astype("float32")-cv2.GaussianBlur(gray.astype("float32"), (0, 0), 3)
        dark = np.maximum(-detail-8, 0).sum()
        light = np.maximum(detail-8, 0).sum()
        polarity = "whiteboard" if dark > light else "blackboard"
    points, colors = [], []
    for y in range(0, h, 32):
        for x in range(0, w, 32):
            tile, g = image[y:y+32, x:x+32], gray[y:y+32, x:x+32]
            lo, hi = np.percentile(g, (60, 90) if polarity == "whiteboard" else (10, 40))
            colors.append(np.median(tile[(g >= lo) & (g <= hi)], axis=0))
            points.append((2*(x+tile.shape[1]/2)/w-1, 2*(y+tile.shape[0]/2)/h-1))
    x, y = np.asarray(points).T
    basis = np.stack((x*0+1, x, y, x*y, x*x, y*y), axis=-1)
    colors = np.asarray(colors)
    # Board illumination can vary smoothly, but a lecturer's clothing must not
    # become the illumination model. Use robust tile chromaticity and a soft
    # brightness prior before fitting, rather than fitting the actor first.
    chroma = colors-colors.mean(axis=1, keepdims=True)
    dominant_chroma = np.median(chroma, axis=0)
    chroma_distance = np.max(np.abs(chroma-dominant_chroma), axis=1)
    tile_gray = colors.mean(axis=1)
    board_level = np.percentile(tile_gray, 75 if polarity == "whiteboard" else 25)
    brightness_distance = np.abs(tile_gray-board_level)
    prior = np.minimum(1, 8/np.maximum(chroma_distance, 1))**2
    prior *= np.minimum(1, 30/np.maximum(brightness_distance, 1))**2
    weights = prior.copy()
    for _ in range(5):
        fit = np.linalg.lstsq(basis*weights[:, None], colors*weights[:, None], rcond=None)[0]
        residual = (colors-basis@fit).mean(axis=1)
        weights = prior*np.minimum(1, 10/np.maximum(np.abs(residual), 1))
        weights[residual < -16 if polarity == "whiteboard" else residual > 16] *= .1
    surface = _surface_basis(h, w) @ fit
    departure = (np.max(np.abs(surface-image), axis=2) > 35).astype("uint8")
    filled = cv2.morphologyEx(departure, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    core = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))
    foreground = cv2.dilate(core, np.ones((9, 9), np.uint8)).astype(bool)
    # A thin forearm can be narrower than the thick foreground core. Extend
    # only substantial skin-colour components connected to existing foreground;
    # this keeps pointed hands out of the confirmed-writing accumulator.
    delta = image.astype(float)-surface
    skin = ((delta[:,:,2]-delta[:,:,0]>12) & (delta[:,:,2]-delta[:,:,1]>4)
            & (np.max(np.abs(delta),axis=2)>20)).astype('uint8')
    skin = cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    skin = cv2.morphologyEx(skin,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
    neighbourhood = cv2.dilate(foreground.astype('uint8'),np.ones((25,25),np.uint8)).astype(bool)
    keep = [i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=20 and np.any(neighbourhood & (labels==i))]
    skin_core=np.isin(labels,keep)
    foreground |= cv2.dilate(skin_core.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
    if return_core:return foreground,surface,core.astype(bool)|skin_core
    return foreground, surface


def measure_features(image, board="auto"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground, surface, foreground_core = surface_foreground(image, None if board == "auto" else board, return_core=True)
    # Polarity of thin local detail survives dark exposure. Absolute brightness
    # is only a low-confidence fallback when no useful marks are visible.
    detail = gray.astype("float32")-cv2.GaussianBlur(gray.astype("float32"), (0, 0), 3)
    dark = float(np.maximum(-detail-8, 0)[~foreground].sum())
    light = float(np.maximum(detail-8, 0)[~foreground].sum())
    confidence = abs(dark-light)/max(1., dark+light)
    if board != "auto":
        kind = board
    elif dark+light > gray.size*.01 and confidence > .12:
        kind = "whiteboard" if dark > light else "blackboard"
    else:
        kind = "whiteboard" if np.median(surface) > 95 else "blackboard"
        confidence = 0.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    op = cv2.MORPH_BLACKHAT if kind == "whiteboard" else cv2.MORPH_TOPHAT
    contrast = cv2.morphologyEx(gray, op, kernel)
    ink = (contrast > 8) & ~foreground
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    sharpness = float(lap[ink].mean()) if ink.any() else 0.
    return dict(ink=ink, foreground=foreground, foreground_core=foreground_core, board=kind, contrast=contrast,
                board_confidence=confidence, dark_detail=dark, light_detail=light,
                surface_median=float(np.median(surface)), ink_sharpness=sharpness)


def features(image, board="auto"):
    m = measure_features(image, board)
    return m["ink"], m["foreground"], m["board"]


def cut_score(a, b):
    delta = np.max(np.abs(a.astype(float)-b), axis=2)
    # Diffuse change across the image; a moving lecturer usually occupies a local region.
    changed = delta > 45
    cells = [c.mean() for row in np.array_split(changed, 4) for c in np.array_split(row, 4, axis=1)]
    return float(changed.mean()), sum(x > 0.4 for x in cells) / 16


def camera_evidence(a, b, foreground_a=None, foreground_b=None):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype("float32")
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype("float32")
    (dx, dy), response = cv2.phaseCorrelate(ga, gb)
    evidence = dict(phase=[float(dx), float(dy), float(response)], verified=False,
                    background_matches=0, inliers=0, reason="no_large_shift")
    if abs(dx) <= 2 and abs(dy) <= 2:
        return False, evidence
    # A moving lecturer can dominate phase correlation. Confirm a distributed
    # background transform before declaring camera motion.
    detector = cv2.ORB_create(nfeatures=1800, fastThreshold=7, edgeThreshold=8, patchSize=21)
    mask_a = None if foreground_a is None else (~foreground_a).astype("uint8")*255
    mask_b = None if foreground_b is None else (~foreground_b).astype("uint8")*255
    ka, da = detector.detectAndCompute(ga.astype("uint8"), mask_a)
    kb, db = detector.detectAndCompute(gb.astype("uint8"), mask_b)
    evidence["reason"] = "insufficient_background_geometry"
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return False, evidence
    matches = [pair[0] for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
               if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
    evidence["background_matches"] = len(matches)
    if len(matches) < 8:
        return False, evidence
    pa = np.float32([ka[x.queryIdx].pt for x in matches])
    pb = np.float32([kb[x.trainIdx].pt for x in matches])
    transform, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.5, maxIters=1000)
    if transform is None:
        return False, evidence
    good = inliers.ravel().astype(bool)
    spread = np.ptp(pa[good], axis=0)/[a.shape[1], a.shape[0]]
    evidence.update(inliers=int(good.sum()), spread=spread.tolist(), transform=transform.tolist())
    if good.sum() < 10 or good.mean() < .5 or spread[0] < .3 or spread[1] < .2:
        return False, evidence
    tx, ty = transform[:, 2]
    scale = float(np.hypot(*transform[:, 0]))
    angle = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    moving = bool(abs(tx) > 2 or abs(ty) > 2 or abs(scale-1) > .01 or abs(angle) > .5)
    evidence.update(verified=True, reason="background_motion" if moving else "foreground_dominated_phase",
                    scale=scale, rotation_degrees=angle)
    return moving, evidence


def camera_shift(a, b):
    moving, evidence = camera_evidence(a, b)
    return moving, evidence["phase"]


def rank_candidates(candidates, observations, shape):
    """Rank observed readable strokes, not a count of arbitrary detected pixels."""
    sample = np.linspace(0, len(candidates)-1, min(96, len(candidates))).astype(int)
    ink_count = np.zeros(shape, np.uint16)
    visible_count = np.zeros(shape, np.uint16)
    def unpack(fid):
        ink, mask = observations[fid]
        return (np.unpackbits(ink, count=int(np.prod(shape))).reshape(shape).astype(bool),
                np.unpackbits(mask, count=int(np.prod(shape))).reshape(shape).astype(bool))
    for idx in sample:
        ink, mask = unpack(candidates[idx]["id"])
        ink_count += ink
        visible_count += ~mask
    support = (ink_count >= min(2, len(sample))) & (ink_count >= .4*visible_count)
    total = int(support.sum())
    ranked = []
    for m in candidates:
        ink, mask = unpack(m["id"])
        near = cv2.dilate(ink.astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
        readable = float((support & near & ~mask).sum()/max(1, total))
        obscured = float((support & mask).sum()/max(1, total))
        score = readable-.8*m["occlusion"]-.15*m["motion_fraction"]
        score += .005*min(1., m["ink_sharpness"]/80)
        parts = dict(score=score, readable_support_fraction=readable,
                     obscured_support_fraction=obscured, support_pixels=total,
                     foreground_fraction=m["occlusion"], motion_fraction=m["motion_fraction"],
                     ink_sharpness=m["ink_sharpness"])
        m["selection"] = parts
        ranked.append(dict(source_frame_id=m["id"], time=m["time"], **parts))
    ranked.sort(key=lambda x:(-x["score"], -x["time"]))
    return ranked


def analyze(root, frames, config, duration):
    """Persistent observed stroke changes form state boundaries, with pre-change preservation.

    Occluded pixels never vote for erasure. Unconfirmed transitions still get their own
    conservative state, preventing speculative cross-state compositing.
    """
    states, metrics = [], []
    previous = None
    previous_foreground = None
    observations = {}
    shape = None

    # Persist the measurements already computed by A so page grouping can
    # reuse them instead of running measure_features() over every frame again.
    feature_folder = feature_cache_dir(root, frames, config)
    feature_folder.mkdir(parents=True, exist_ok=True)
    feature_inks, feature_masks, feature_sharpness = [], [], []
    feature_start = 0

    def flush_feature_chunk():
        nonlocal feature_start
        if not feature_inks:
            return
        write_feature_chunk(
            feature_folder,
            feature_start,
            feature_inks,
            feature_masks,
            feature_sharpness,
            shape,
        )
        feature_start += len(feature_inks)
        feature_inks.clear()
        feature_masks.clear()
        feature_sharpness.clear()
    known = reference = streak = None
    pending_start = None
    state_start = scene = 0
    reason = "initial"
    current_flags = set()

    def close(end_index, end_time, why):
        nonlocal state_start, current_flags
        if end_index <= state_start:
            return
        ids = [f["id"] for f in frames[state_start:end_index]]
        candidates = metrics[state_start:end_index]
        ranking = rank_candidates(candidates, observations, shape)
        best = ranking[0]
        states.append(dict(id=f"b{len(states):05d}", scene=scene,
                           start=frames[state_start]["time"], end=end_time,
                           source_frame_ids=ids, selected_frame_id=best["source_frame_id"],
                           selection_reason="persistent_visible_strokes_foreground_and_stability",
                           selection_candidates=ranking[:5],
                           boundary_reason=why, quality_flags=sorted(current_flags),
                           transcript_ids=[]))
        for fid in ids:
            observations.pop(fid, None)
        state_start = end_index
        current_flags = set()

    for i, f in enumerate(frames):
        raw = cv2.imread(str(root / f["image"]))
        image = small(raw, config)
        measured = measure_features(image, config.board)
        ink, foreground, kind = measured["ink"], measured["foreground"], measured["board"]
        shape = ink.shape
        packed_ink = np.packbits(ink)
        packed_foreground = np.packbits(foreground)
        observations[f["id"]] = (packed_ink, packed_foreground)
        feature_inks.append(packed_ink)
        feature_masks.append(packed_foreground)
        feature_sharpness.append(measured["ink_sharpness"])
        if len(feature_inks) >= FEATURE_CHUNK_SIZE or i + 1 == len(frames):
            flush_feature_chunk()

        cut, distribution = cut_score(previous, image) if previous is not None else (0., 0.)
        moving, camera = camera_evidence(previous, image, previous_foreground, foreground) if previous is not None else (False, dict(phase=[0,0,1], verified=True, reason="initial"))
        motion = float((np.max(np.abs(image.astype(float)-previous), axis=2) > 20).mean()) if previous is not None else 0.
        scene_cut = cut > config.cut_fraction and distribution > .7
        m = dict(id=f["id"], time=f["time"], occlusion=float(foreground.mean()),
                 sharpness=float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
                 ink_pixels=int(ink.sum()), cut=cut, shift=camera["phase"], camera=camera, board=kind,
                 board_confidence=measured["board_confidence"], dark_detail=measured["dark_detail"],
                 light_detail=measured["light_detail"], surface_median=measured["surface_median"],
                 ink_sharpness=measured["ink_sharpness"], motion_fraction=motion,
                 additions=0, removals=0, event="none")
        metrics.append(m)
        if scene_cut or moving:
            if pending_start is not None and pending_start > state_start:
                close(pending_start, frames[pending_start]["time"], "uncertain_transition")
                current_flags.add("unconfirmed_change")
            close(i, f["time"], "camera_motion" if moving and not scene_cut else "scene_cut")
            scene += 1
            known = reference = streak = None
            pending_start = None
            if moving and not scene_cut:
                current_flags.add("camera_motion_reconstruction_disabled")
            m["event"] = "scene_cut" if scene_cut else "camera_motion"
        if reference is None:
            reference = ink.copy()
            known = ~foreground
            streak = np.zeros(ink.shape, np.uint16)
        else:
            visible = ~foreground
            newly_known = visible & ~known
            reference[newly_known] = ink[newly_known]
            known |= visible
            # One analysis pixel of edge jitter is common after video compression.
            # Count genuinely new/disappeared strokes, rather than shifted boundaries.
            near_reference = cv2.dilate(reference.astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
            near_ink = cv2.dilate(ink.astype("uint8"), np.ones((3, 3), np.uint8)).astype(bool)
            disagreement = visible & known & ((ink & ~near_reference) | (reference & ~near_ink))
            streak = np.where(disagreement, streak + 1, 0).astype(np.uint16)
            persistent = streak >= config.confirm
            adds = int((persistent & ink).sum())
            removes = int((persistent & ~ink).sum())
            m.update(additions=adds, removals=removes)
            threshold = max(config.change_pixels, int(ink.size * config.change_fraction))
            if disagreement.sum() >= threshold and pending_start is None:
                pending_start = i
            if adds + removes >= threshold:
                # Confirmed pixels have a measured consecutive onset. An old,
                # unrelated pending fluctuation must not backdate this boundary.
                boundary = max(state_start, i-config.confirm+1)
                why = "erasure_or_revision" if removes >= threshold else "writing_progress"
                close(boundary, frames[boundary]["time"], why)
                reference[visible] = ink[visible]
                streak[:] = 0
                pending_start = None
                m["event"] = why
            elif pending_start is not None and not disagreement.any():
                # Short-lived writing is retained even if it never became persistent.
                close(pending_start, frames[pending_start]["time"], "transient_change")
                current_flags.add("unconfirmed_change")
                close(i, f["time"], "transient_return")
                pending_start = None
        if foreground.mean() > .03:
            current_flags.add("foreground_heuristic")
        previous = image
        previous_foreground = foreground
    if pending_start is not None and pending_start > state_start:
        close(pending_start, frames[pending_start]["time"], "unconfirmed_change")
        current_flags.add("unconfirmed_change")
    close(len(frames), duration, "end_of_video")
    return dict(states=states, metrics=metrics)
