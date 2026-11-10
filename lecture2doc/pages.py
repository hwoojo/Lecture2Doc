"""Accumulated board pages. GT is never consumed; output pixels are source pixels."""
import cv2
import numpy as np
from .vision import small, measure_features
from .feature_cache import load_feature_observations
from .storage import write, read, key
import os

PAGE_VERSION = '2026-11-09.pages-18-shared-features'


def observations(root, folder, frames, config, progress=None):
    """Load A's shared measurements, falling back to the legacy page cache."""
    shared = load_feature_observations(root, frames, config)
    if shared is not None:
        if progress:
            progress(f"공유 판서 특징 캐시 재사용 {len(frames)}/{len(frames)}")
        return shared

    previous = folder
    folder = root / 'cache' / 'page-observations' / key([frames,config.board,config.analysis_width,config.roi], 'page-measurements-3')
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(frames), 128):
        path = folder / f'features-{start:06d}.npz'
        if not path.exists():
            inks, masks, sharp = [], [], []
            for f in frames[start:start+128]:
                image = cv2.imread(str(root / f['image']))
                if image is None:
                    raise ValueError(f"Unreadable source {f['id']}")
                m = measure_features(small(image, config), config.board)
                inks.append(np.packbits(m['ink']))
                masks.append(np.packbits(m['foreground']))
                sharp.append(m['ink_sharpness'])
                shape = m['ink'].shape
            temp=path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, ink=inks, mask=masks, sharp=sharp, shape=shape)
            os.replace(temp,path)
        with np.load(path) as data:
            chunks.append({k: data[k].copy() for k in data.files})
        if progress:
            progress(f"완성 판서 관측 {min(start+128,len(frames))}/{len(frames)}")
    if not chunks:
        return None
    return dict(ink=np.concatenate([c['ink'] for c in chunks]),
                mask=np.concatenate([c['mask'] for c in chunks]),
                sharp=np.concatenate([c['sharp'] for c in chunks]), shape=tuple(chunks[0]['shape']))


def persistent_border_structures(ink_frequency,foreground_frequency,bounds):
    """Exclude observed static border furniture, not a frequently standing actor."""
    h,w=ink_frequency.shape;x0,y0,x1,y1=bounds;bw,bh=x1-x0,y1-y0
    output=np.zeros((h,w),bool)
    visible_frequency=np.maximum(1-foreground_frequency,.01)
    stable_ink=(ink_frequency/visible_frequency>=.85)&(ink_frequency>=.2)
    for kind,stable in [('ink',stable_ink),('foreground',foreground_frequency>=.95)]:
        joined=cv2.morphologyEx(stable.astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(joined)
        for i in range(1,count):
            x,y,cw,ch,area=map(int,stats[i])
            horizontal=(cw>bw*.15 and cw/ch>=15 and 4<=ch<bh*.09 and y+ch>y1-bh*.09)
            vertical=(ch>bh*.15 and 3<=cw<bw*.05 and (x<x0+bw*.04 or x+cw>x1-bw*.04))
            border=x<x0+bw*.04 or x+cw>x1-bw*.04 or y<y0+bh*.04 or y+ch>y1-bh*.06
            thin_static=kind=='foreground' and border and (area<bw*bh*.015 or max(cw/ch,ch/cw)>6)
            if horizontal or vertical or thin_static:
                output[max(0,y-2):min(h,y+ch+2),max(0,x-2):min(w,x+cw+2)]=True
    return output


def localized_loss_region(removed,reference):
    """A substantial local stroke loss must not be diluted by the whole board."""
    minimum=max(12,int(removed.size*.00005))
    if int(removed.sum())<minimum:return None
    horizontal=max(3,int(round(removed.shape[1]*.025))|1)
    joined=cv2.dilate(removed.astype('uint8'),np.ones((5,horizontal),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined)
    hits=np.bincount(labels[removed],minlength=count)
    for i in np.argsort(hits[1:])[::-1]+1:
        if hits[i]<minimum:break
        yy,xx=np.nonzero(removed & (labels==i));x,y=int(xx.min()),int(yy.min());w,h=int(xx.max()-x+1),int(yy.max()-y+1)
        x0,y0=max(0,x-3),max(0,y-3);x1,y1=min(reference.shape[1],x+w+3),min(reference.shape[0],y+h+3)
        before=(reference|removed)[y0:y1,x0:x1]
        fraction=float(hits[i]/max(1,before.sum()))
        if fraction>=.5:
            return dict(mask=removed&(labels==i),pixels=int(hits[i]),fraction=fraction,box=[x0,y0,x1,y1])
    return None


def group_pages(frames, data, config, duration, metrics=(), diagnostic=None):
    """Accumulate additions; split only on sustained, visible loss of old writing.

    Missing pixels underneath foreground never vote for erasure. Deletion sweeps
    have a settling period so each partly wiped frame cannot create a page.
    """
    if not frames:
        return dict(pages=[], events=[])
    data=dict(data,ink=data['ink'].copy(),mask=data['mask'].copy())
    alignment_events=[]; aligned_ids=set();alignment_steps=[];epoch_start=0
    shape = tuple(map(int,data['shape'])); n = int(np.prod(shape)); kernel = np.ones((3,3),np.uint8)
    def unpack(kind,i):
        return np.unpackbits(data[kind][i],count=n).reshape(shape).astype(bool)
    def near(a):
        return cv2.dilate(a.astype('uint8'),kernel).astype(bool)
    metric_by_id={m['id']:m for m in metrics}
    for i,f in enumerate(frames):
        metric=metric_by_id.get(f['id'],{}); camera=metric.get('camera',{})
        matrix=camera.get('transform')
        if metric.get('event')=='camera_motion' and camera.get('verified') and matrix:
            mat=np.asarray(matrix,dtype='float32')
            if (abs(mat[0,2])<shape[1]*.05 and abs(mat[1,2])<shape[0]*.05
                and abs(camera.get('scale',1)-1)<.03 and abs(camera.get('rotation_degrees',0))<2):
                for j in range(epoch_start,i):
                    for kind in ('ink','mask'):
                        warped=cv2.warpAffine(unpack(kind,j).astype('uint8'),mat,(shape[1],shape[0]),
                            flags=cv2.INTER_NEAREST,borderValue=1 if kind=='mask' else 0)
                        data[kind][j]=np.packbits(warped)
                alignment_steps.append((epoch_start,i,mat))
                aligned_ids.add(f['id'])
                alignment_events.append(dict(time=f['time'],kind='analysis_camera_alignment',transform=matrix,
                                             output_pixels_unchanged=True))
        if metric.get('event') in ('camera_motion','scene_cut') and f['id'] not in aligned_ids:
            epoch_start=i
    # Static frame/rail components at the image edge are not inscriptions.
    # Derive exclusions from source observations, never from annotation masks.
    sample=np.linspace(0,len(frames)-1,min(64,len(frames)),dtype=int)
    frequency=sum(unpack('ink',int(j)).astype('float32') for j in sample)/len(sample)
    fixtures=np.zeros(shape,bool);h,w=shape
    # A pair of long, persistent opposite rails provides a source-only board
    # rectangle. Without that evidence, retain the configured ROI in full.
    edge_support=frequency>.2
    rows=edge_support.sum(1);cols=edge_support.sum(0)
    top=np.flatnonzero((np.arange(h)<h*.2)&(rows>w*.45))
    bottom=np.flatnonzero((np.arange(h)>h*.8)&(rows>w*.45))
    left=np.flatnonzero((np.arange(w)<w*.2)&(cols>h*.45))
    right=np.flatnonzero((np.arange(w)>w*.8)&(cols>h*.45))
    bounds=[0,0,w,h]
    if len(top) and len(bottom):
        bounds[1]=int(top.max()+3);bounds[3]=int(bottom.min()-2)
    if len(left) and len(right):
        bounds[0]=int(left.max()+3);bounds[2]=int(right.min()-2)
    xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    fixtures |= (xx<bounds[0])|(xx>=bounds[2])|(yy<bounds[1])|(yy>=bounds[3])
    foreground_frequency=sum(unpack('mask',int(j)).astype('float32') for j in sample)/len(sample)
    fixtures |= persistent_border_structures(frequency,foreground_frequency,bounds)
    x0,y0,x1,y1=bounds;bw,bh=x1-x0,y1-y0
    border_band=(xx<x0+bw*.035)|(xx>x1-bw*.035)|(yy<y0+bh*.04)|(yy>y1-bh*.08)
    # Static furniture can alternate between the foreground and ink labels.
    # Keep this exclusion at the board perimeter, never over the whole lecturer.
    fixtures |= cv2.dilate((border_band & (foreground_frequency>.5)).astype('uint8'),np.ones((9,9),np.uint8)).astype(bool)
    persistent_unknown=(foreground_frequency>=.95)&~fixtures
    unknown_area=float(persistent_unknown.mean())
    fixtures=near(fixtures)
    fixtures_for=lambda i:fixtures
    raw_unpack=unpack
    def unpack(kind,i):
        value=raw_unpack(kind,i)
        mask=cv2.dilate(raw_unpack('mask',i).astype('uint8'),np.ones((7,7),np.uint8)).astype(bool)
        return (value & ~mask & ~fixtures) if kind=='ink' else (mask & ~fixtures)
    times = np.array([f['time'] for f in frames])
    # Counts express elapsed observation time, including with non-default sampling.
    confirm = max(2, int(np.ceil(3.0/config.sample_seconds)))
    absence = max(3, int(np.ceil(2.0/config.sample_seconds)))
    stable = np.zeros(shape,np.uint16); missing=stable.copy()
    reference=np.zeros(shape,bool)
    blank_streak=np.zeros(shape,np.uint16);last_blank=np.full(shape,-1,np.int32);last_positive=np.full(shape,-1,np.int32);evidence_history={}
    frame_indices={f['id']:i for i,f in enumerate(frames)}
    history={};activity_history={};last_activity=np.full(shape,-1,np.int32);last_visible_ink=np.zeros(shape,bool);logged_corrections=set()
    last_page_target=None
    pages=[]; events=list(alignment_events); start=0; erasing=False; last_loss=-1e9; scene=0; pending_loss=0; pending_target=None; pending_onset=0; pending_started=-1e9; pending_removed=np.zeros(shape,bool);pending_onsets=np.full(shape,-1,np.int32);pending_visible_steps=np.zeros(shape,np.uint16)
    by_metric={m['id']:m for m in metrics}
    def emit(stop,target,reason):
        nonlocal start,last_page_target
        if stop <= start:return
        original_target=target
        blank_at_stop,positive_at_stop=evidence_history.get(stop,(last_blank,last_positive))
        recent_votes=sum(unpack('ink',j).astype('uint8') for j in range(max(start,stop-4),stop))
        provisional=(recent_votes>=2)&(positive_at_stop>blank_at_stop)&~fixtures
        new_provisional=int((provisional & ~near(target)).sum())
        target=target|provisional
        if int(target.sum())<100:return
        candidates=[]
        for j in range(start,stop):
            ink,mask=unpack('ink',j),unpack('mask',j)
            readable=float((target & near(ink) & ~mask).sum()/max(1,target.sum()))
            hidden=float((target & mask).sum()/max(1,target.sum()))
            extra=float((ink & ~near(target)).sum()/max(1,target.sum()))
            score=readable-.6*extra-.15*hidden
            candidates.append(dict(source_frame_id=frames[j]['id'],time=float(times[j]),score=score,
                                   readable_fraction=readable,hidden_fraction=hidden,extra_fraction=extra,
                                   foreground_fraction=float(mask.mean()),ink_sharpness=float(data['sharp'][j])))
        candidates.sort(key=lambda c:(c['score'],c['readable_fraction'],-c['hidden_fraction'],-c['foreground_fraction'],c['ink_sharpness'],c['time']),reverse=True)
        # Threshold/raster noise of at most five analysis pixels is not a
        # reason to prefer a lecturer entering an otherwise complete board.
        # Keep both content and score constraints so a newly added symbol
        # cannot be traded for a globally smaller foreground mask.
        leader=candidates[0];tolerance=5/max(1,int(target.sum()))
        equivalent=[c for c in candidates if c['score']>=leader['score']-tolerance-1e-12
                    and c['readable_fraction']>=leader['readable_fraction']-tolerance-1e-12
                    and c['hidden_fraction']<=leader['hidden_fraction']+tolerance+1e-12]
        equivalent.sort(key=lambda c:(-c['foreground_fraction'],c['score'],c['ink_sharpness'],c['time']),reverse=True)
        chosen_ids={c['source_frame_id'] for c in equivalent}
        candidates=equivalent+[c for c in candidates if c['source_frame_id'] not in chosen_ids]
        best=candidates[0]
        if reason=='confirmed_erasure_or_revision' and last_page_target is not None:
            novel_mask=target & ~near(last_page_target)
            count,labels,stats,_=cv2.connectedComponentsWithStats(novel_mask.astype('uint8'))
            # Isolated threshold flecks around erased strokes and the frame
            # cannot turn a partly wiped board into another completed page.
            discernible=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=5]
            novel=int(np.isin(labels,discernible).sum())
            if novel<max(5,int(target.size*.00002)):
                events.append(dict(time=float(times[stop]),kind='erasure_continuation_not_a_completed_page',
                    novel_pixels=novel,raw_novel_pixels=int(novel_mask.sum()),best_readable_fraction=best['readable_fraction']))
                # A suppressed wipe belongs to the preceding board, not the
                # next topic. Otherwise its timestamps and donors would be
                # silently assigned to a future, incompatible board.
                previous=pages[-1]
                previous['end']=float(times[stop])
                previous['source_frame_ids'].extend(f['id'] for f in frames[start:stop])
                previous.setdefault('suppressed_erasure_intervals',[]).append(dict(start=float(times[start]),end=float(times[stop]),reason='no_discernible_new_writing'))
                start=stop
                if diagnostic:
                    path=diagnostic/(previous['id']+'.candidates.json')
                    saved=read(path);saved['page']=previous;write(path,saved)
                return False
        flags=[]
        if new_provisional>=8:flags.append('provisional_terminal_writing')
        if original_target.sum()<100:flags.append('insufficient_stable_observations')
        if best['readable_fraction']<.97:
            flags.append('complete_original_not_observed')
        if unknown_area>.03:
            flags.append('persistent_foreground_unobserved')
        if best['foreground_fraction']>.03:
            flags.append('foreground_present_original_used')
        if best['extra_fraction']>.05:
            flags.append('terminal_content_uncertain')
        selected_index=frame_indices[best['source_frame_id']]
        transform=np.eye(3,dtype='float32')
        for epoch,step_index,step in alignment_steps:
            if epoch<=selected_index<step_index:
                transform=np.vstack([step,[0,0,1]])@transform
        inverse=cv2.invertAffineTransform(transform[:2].astype('float32'))
        native_target=cv2.warpAffine(target.astype('uint8'),inverse,(shape[1],shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
        native_board=cv2.warpAffine((~fixtures_for(selected_index)).astype('uint8'),inverse,(shape[1],shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
        page=dict(id=f'p{len(pages):05d}',scene=scene,start=float(times[start]),
                  end=float(times[stop]) if stop<len(frames) else duration,
                  source_frame_ids=[f['id'] for f in frames[start:stop]],
                  selected_frame_id=best['source_frame_id'],selection_reason='terminal_readable_writing',
                  selection_tolerance_ink_pixels=5,content_equivalent_candidate_ids=[c['source_frame_id'] for c in equivalent],
                  selection_candidates=candidates[:8],boundary_reason=reason,
                  quality_flags=flags,transcript_ids=[],target_ink_pixels=int(target.sum()),
                  original_complete_estimate=not flags,provisional_ink_pixels=new_provisional,terminal_target_coordinate_system='selected_frame_analysis_roi',analysis_to_selected_transform=inverse.tolist())
        pages.append(page)
        last_page_target=target.copy()
        if diagnostic:
            write(diagnostic/(page['id']+'.candidates.json'),dict(page=page,candidates=sorted(candidates,key=lambda c:c['time'])))
            cv2.imwrite(str(diagnostic/(page['id']+'.target.png')),native_target.astype('uint8')*255)
            cv2.imwrite(str(diagnostic/(page['id']+'.board.png')),native_board.astype('uint8')*255)
            blank,positive=evidence_history.get(stop,(last_blank,last_positive))
            def stamp_seconds(indices):
                values=np.full(shape,-1,dtype='float32');known=indices>=0
                values[known]=times[indices[known]]
                return cv2.warpAffine(values,inverse,(shape[1],shape[0]),flags=cv2.INTER_NEAREST,borderValue=-1)
            np.savez_compressed(diagnostic/(page['id']+'.evidence.npz'),last_blank_seconds=stamp_seconds(blank),last_positive_seconds=stamp_seconds(positive))
        return True
    for i,f in enumerate(frames):
        ink,mask=unpack('ink',i),unpack('mask',i)
        metric=by_metric.get(f['id'],{})
        # A verified geometric discontinuity invalidates pixel correspondence.
        discontinuity=metric.get('event') in ('scene_cut','camera_motion') and f['id'] not in aligned_ids
        if discontinuity and i>start:
            emit(i,reference.copy(),'camera_or_scene_change')
            events.append(dict(time=float(times[i]),kind='camera_or_scene_change'))
            reference[:]=False;stable[:]=0;missing[:]=0;start=i;scene+=1;erasing=False;pending_loss=0;pending_target=None;pending_removed[:]=False;history.clear();activity_history.clear();last_activity[:]=-1;last_visible_ink[:]=False;blank_streak[:]=0;last_blank[:]=-1;last_positive[:]=-1;evidence_history.clear();last_page_target=None
        visible=~mask
        stable=np.where(ink, np.minimum(stable+1,60000),np.where(mask,stable,0)).astype('uint16')
        confirmed=stable>=confirm
        evidence_history[i]=(last_blank.copy(),last_positive.copy())
        evidence_history.pop(i-absence-int(4/config.sample_seconds)-confirm-2,None)
        near_ink=near(ink)
        blank_streak=np.where(visible & ~near_ink,np.minimum(blank_streak+1,60000),0).astype('uint16')
        last_blank[blank_streak>=absence]=i
        last_positive[ink]=i
        activity_history[i]=last_activity.copy()
        new_evidence=ink & ~near(last_visible_ink) & ~near(reference)
        if new_evidence.sum()>=3:
            count,labels,stats,_=cv2.connectedComponentsWithStats(new_evidence.astype('uint8'))
            keep=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]>=3]
            last_activity[np.isin(labels,keep)]=i
        last_visible_ink[visible]=ink[visible]
        old=reference.copy()
        activity_history.pop(i-absence-int(4/config.sample_seconds)-confirm-2,None)
        history[i]=old
        history.pop(i-absence-int(4/config.sample_seconds)-confirm-2,None)
        missing=np.where(visible & reference & ~near(ink),np.minimum(missing+1,60000),0).astype('uint16')
        removed=missing>=absence
        loss=int(removed.sum())
        threshold=max(60,int(old.sum()*.08))
        if times[i]-pending_started>4:
            pending_loss=0;pending_target=None;pending_removed[:]=False;pending_onsets[:]=-1
        if loss:
            if pending_target is None:
                pending_started=float(times[i]);pending_onset=max(start+1,i-absence+1)
                pending_target=history.get(pending_onset,old).copy()
            pending_visible_steps[removed & ~pending_removed]=absence-1
            pending_onsets[removed & ~pending_removed]=i-absence+1
            pending_removed |= removed
            pending_loss=int(pending_removed.sum())
        if erasing and pending_loss>=threshold*.25:
            last_loss=float(times[i])
        returned=pending_removed & visible & near(ink)
        pending_removed[returned]=False
        pending_visible_steps=np.where(pending_removed & visible,np.minimum(pending_visible_steps+1,60000),0).astype('uint16')
        pending_loss=int(pending_removed.sum())
        eligible_local=pending_removed & (pending_visible_steps>=max(absence,int(np.ceil(3/config.sample_seconds))))
        local=None
        for _ in range(8):
            candidate=localized_loss_region(eligible_local,old)
            if candidate is None:break
            first=max(start+1,int(pending_onsets[candidate['mask']].min()))
            x0,y0,x1,y1=candidate['box'];padding=max(8,y1-y0);horizontal_padding=max(16,2*(y1-y0))
            timestamps=activity_history.get(first,last_activity)[max(0,y0-padding):min(shape[0],y1+padding),max(0,x0-horizontal_padding):min(shape[1],x1+horizontal_padding)]
            recent=int(timestamps.max(initial=-1))
            quiet=float(times[first]-times[recent]) if recent>=0 else float('inf')
            if quiet>=6:
                local=candidate;local['quiet_seconds']=quiet if np.isfinite(quiet) else None;break
            marker=(first//confirm,x0//8,y0//8,x1//8,y1//8)
            if marker not in logged_corrections:
                events.append(dict(time=float(times[i]),onset=float(times[first]),kind='in_progress_correction',box=candidate['box'],pixels=candidate['pixels'],quiet_seconds=quiet))
                logged_corrections.add(marker)
            eligible_local &= ~candidate['mask']
        global_major=pending_loss>=max(threshold,int(n*.002))
        if global_major or local is not None:
            onset=pending_onset
            target_before=pending_target
            if local is not None:
                onset=max(start+1,int(pending_onsets[local['mask']].min()))
                target_before=history.get(onset,pending_target)
            if not erasing:
                emitted=emit(onset,target_before,'confirmed_erasure_or_revision')
                if emitted:
                    start=onset
                erasing=True
            events.append(dict(time=float(times[i]),onset=float(times[onset]),kind='visible_writing_loss',
                               removed_pixels=pending_loss,reference_pixels=int(old.sum()),local_loss={k:v for k,v in local.items() if k!='mask'} if local else None))
            last_loss=float(times[i]);pending_loss=0;pending_target=None;pending_removed[:]=False;pending_onsets[:]=-1
        reference |= confirmed
        reference[removed]=False
        missing[removed]=0
        if erasing and times[i]-last_loss>=4:
            erasing=False
        # Do not erase a small correction silently: log losses below page threshold.
        if loss and loss<threshold and loss>=20:
            events.append(dict(time=float(times[i]),kind='minor_loss',removed_pixels=loss,threshold=threshold))
    emit(len(frames),reference,'end_of_video')
    if not pages:
        # A short or heavily occluded clip must still retain an observed source.
        counts=[int(unpack('ink',j).sum()) for j in range(len(frames))]
        if max(counts,default=0)>=100:
            j=max(range(len(frames)),key=lambda j:(counts[j],times[j]))
            emit(len(frames),unpack('ink',j),'insufficient_stable_observations')
            pages[-1]['quality_flags'].append('insufficient_stable_observations')
            pages[-1]['original_complete_estimate']=False
    for page in pages:
        page['writing_loss_barriers']=sorted({e['onset'] for e in events if e['kind'] in ('visible_writing_loss','in_progress_correction') and page['start']<=e['onset']<page['end']})
    return dict(pages=pages,events=events,policy=dict(version=PAGE_VERSION,
        addition_confirmation_seconds=3,visible_absence_seconds=2,erasure_fraction=.08,
        minimum_erasure_pixels=60,localized_erasure_minimum=max(12,int(n*.00005)),localized_absence_seconds=3,localized_min_quiet_seconds=6,localized_erasure_fraction=.5,erasure_settle_seconds=4,uses_ground_truth=False,
        output_pixels='selected_original_only',source_inferred_board_bounds=bounds,persistent_unknown_fraction=unknown_area))


def build_pages(root,folder,frames,config,duration,metrics,progress=None):
    data=observations(root,folder,frames,config,progress)
    result=group_pages(frames,data,config,duration,metrics,folder) if data else dict(pages=[],events=[])
    for page in result['pages']:
        for field,suffix in [('terminal_ink_mask','.target.png'),('analysis_board_mask','.board.png'),('terminal_absence_evidence','.evidence.npz')]:
            page[field]=(folder/(page['id']+suffix)).relative_to(root).as_posix()
    return result
