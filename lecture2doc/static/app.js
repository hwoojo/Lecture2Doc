const $=id=>document.getElementById(id);
const records=()=>manifest.board_pages??manifest.board_states;
const pageBAvailable=()=>manifest.page_reconstruction_status==='complete'&&manifest.page_reconstructions?.length;
let currentId=null, manifest=null, currentState=0, poll=null;
const selectedBoards=new Set();
const boardNumber=(s,i)=>s.display_number??i+1;
const editOptions=(method,body)=>({method,headers:{'X-Lecture2Doc':'1','Content-Type':'application/json'},...(body?{body:JSON.stringify(body)}:{})});
let libraryLoading=false,libraryRefreshRequested=false,hasActiveJobs=false,lastJobsKey=null,lastTrashKey=null;
const statusLabels={queued:'대기 중',running:'처리 중',complete:'완료',failed:'실패',interrupted:'중단됨'};
function syncElapsed(label,status){
 if(Number.isFinite(status.elapsed_seconds)){
  label.dataset.elapsedBase=String(status.elapsed_seconds);label.dataset.measuredAt=String(performance.now());
  label.dataset.ticking=String(['running','queued'].includes(status.status));label.dataset.label=status.status==='queued'?'대기':'경과';
 }
}
function elapsedLabel(status){const label=text('small','','elapsed-time');syncElapsed(label,status);return label}
function confirmPurge(job){
 const dialog=$('purge-confirm');$('purge-name').textContent=job.name;
 $('purge-scope').textContent=job.purge_scope==='library_entry'?'이 강의를 목록에서 영구 삭제합니다. 외부 원본 폴더는 유지됩니다.':'이 강의의 원본 영상·판서·전사·분석 파일을 모두 삭제합니다.';
 dialog.returnValue='cancel';
 return new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='purge'),{once:true});dialog.showModal()})
}
function updateElapsedClocks(){document.querySelectorAll('[data-elapsed-base]').forEach(e=>{
 const seconds=Number(e.dataset.elapsedBase)+(e.dataset.ticking==='true'?(performance.now()-Number(e.dataset.measuredAt))/1000:0);
 e.textContent=`${e.dataset.label} ${fmt(Math.max(0,seconds))}`;
})}
function showProcessingStatus(status){$('upload-status').replaceChildren(text('span',status.message),elapsedLabel(status));updateElapsedClocks()}
setInterval(()=>{updateElapsedClocks();if(hasActiveJobs&&!$('library').hidden)library()},1000);
const notice=message=>{$('action-status').textContent=message};
const labels={no_substantial_board_occluder:'칠판 내부의 뚜렷한 가림이 없어 원본 유지',provisional_terminal_writing:'구간 끝의 필기 포함',no_observed_hidden_writing:'다른 시점에서 확인된 숨은 필기가 없음',fixed_camera_and_board_roi_require_review:'고정 카메라·판서 영역을 확인하세요',sampling_can_miss_short_lived_writing:'분석 간격보다 짧게 나타나는 판서는 누락될 수 있습니다',persistence_can_miss_short_lived_writing:'3초 이내에 추가·삭제되는 필기는 누락될 수 있습니다',input_below_720p:'720p 미만 입력',foreground_heuristic:'가림 추정 사용',unconfirmed_change:'확인되지 않은 변화',camera_motion_reconstruction_disabled:'카메라 이동 · 복원 제한',model_free_masks_require_review:'가림 추정',unsafe_state_original_used:'불확실한 상태 · 원본 제공',insufficient_observations:'관측 프레임 부족',inconsistent_writing_original_used:'판서 불일치 · 원본 제공',unverified_regions_original_retained:'미확인 영역은 선택 원본 유지',no_verified_improvement:'검증된 개선 없음 · 원본 제공',transcript_failed:'음성 전사 실패',reconstruction_disabled:'복원 비활성화',writing_progress:'판서 추가',erasure_or_revision:'삭제·수정',scene_cut:'장면 전환',end_of_video:'영상 끝',initial:'시작',clear_original_sufficient:'원본 하나로 충분함',clear_compatible_original_preferred:'동일 내용의 깨끗한 원본을 우선 사용',observed_hidden_writing_recovered:'다른 원본에서 반복 관측한 필기를 복원',insufficient_same_state_observations:'같은 판서 상태의 원본 부족',insufficient_compatible_observations:'내용·정렬이 일치하는 원본 부족',no_repeatedly_observed_hidden_writing:'다른 원본에서 가린 필기를 확인하지 못함',no_safe_hidden_writing_gain:'안전하게 추가 보존할 필기 부족',visible_writing_loss_original_used:'기존 필기 소실 우려로 합성 취소',foreground_residual_original_used:'합성 영역에 전경 잔상 우려로 취소',hidden_writing_not_preserved:'가린 필기 보존 확인 실패',unsafe_terminal_state:'판서 상태가 불확실해 복원 보류',complete_original_not_observed:'A 원본에는 최종 판서를 모두 드러낸 장면이 없음',insufficient_stable_observations:'필기를 확인할 안정된 관측이 부족함 · 원본 후보 제공',persistent_foreground_unobserved:'자주 가려지는 영역의 완전성 미확인',foreground_present_original_used:'A 선택 원본에 전경 가림 가능 영역이 있음',source_illumination_adjusted:'원본의 빈 칠판 관측으로 밝기 보정 · 계수는 출처 JSON에 기록',heuristic_restoration_requires_review:'관측 영역 합성',unobserved_regions_original_retained:'미확인 가림 영역은 A 원본 유지',page_restoration_failed:'복원 오류 · A 원본 유지',terminal_state_unavailable:'동일 판서 구간 확인 불가 · 원본 유지',terminal_target_geometry_mismatch:'판서 좌표 확인 불가 · 원본 유지',terminal_content_uncertain:'삭제 직전 판서 내용 추정에 불확실성 있음'};
const fmt=t=>`${Math.floor(t/3600)?Math.floor(t/3600)+':':''}${String(Math.floor(t/60)%60).padStart(2,'0')}:${String(Math.floor(t)%60).padStart(2,'0')}`;
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e};
const file=p=>`/api/jobs/${currentId}/files/${p.split('/').map(encodeURIComponent).join('/')}`;
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let x=await r.json().catch(()=>({detail:r.statusText}));throw new Error(typeof x.detail==='string'?x.detail:JSON.stringify(x.detail))}return r.json()}
async function library(force=false){if(libraryLoading){if(force)libraryRefreshRequested=true;return}libraryLoading=true;try{
 const jobs=await api('/api/jobs');hasActiveJobs=jobs.some(j=>['running','queued'].includes(j.status));const jobsKey=JSON.stringify(jobs.map(j=>[j.id,j.name,j.status,j.message,j.can_delete]));if(jobsKey!==lastJobsKey){lastJobsKey=jobsKey;$('count').textContent=jobs.length;$('jobs').replaceChildren();
 if(!jobs.length)$('jobs').append(text('p','등록한 강의가 없습니다. 첫 강의를 추가해보세요.','hint'));
 for(const job of jobs){
  const row=text('div','','job'),info=text('div',''),actions=text('div','','job-actions');row.dataset.jobId=job.id;
  info.append(text('p',job.name||`강의 ${job.id.slice(0,6)}`),text('small',job.message===(statusLabels[job.status]||job.status)?job.message:`${statusLabels[job.status]||job.status} · ${job.message}`),elapsedLabel(job));row.append(info);
  if(job.status==='complete'){const b=text('button','기록 열기 →');b.onclick=()=>openJob(job.id);actions.append(b)}
  const remove=text('button','삭제','danger');remove.setAttribute('aria-label',`${job.name||'강의'} 삭제`);remove.disabled=job.can_delete===false;remove.onclick=()=>deleteLecture(job.id);actions.append(remove);row.append(actions);$('jobs').append(row);
 }
 }else{for(const job of jobs){const timer=document.querySelector(`[data-job-id="${job.id}"] .elapsed-time`);if(timer)syncElapsed(timer,job)}}
 const trash=await api('/api/trash');const trashKey=JSON.stringify(trash);if(trashKey!==lastTrashKey){lastTrashKey=trashKey;$('trash-count').textContent=`(${trash.length})`;$('trash-jobs').replaceChildren();
 for(const job of trash){
  const row=text('div','','trash-row'),restore=text('button','복원'),purge=text('button','영구 삭제','danger'),actions=text('div','','job-actions');row.append(text('span',job.name));
  restore.onclick=async()=>{try{await api(`/api/jobs/${job.id}/restore`,editOptions('POST'));notice('강의를 복원했습니다.');await library(true)}catch(e){notice(e.message)}};
  purge.onclick=async()=>{
   if(!await confirmPurge(job))return;
   purge.disabled=true;restore.disabled=true;purge.textContent='삭제 중…';
   try{await api(`/api/jobs/${job.id}/purge`,editOptions('POST',{confirmed:true}));notice('강의를 영구 삭제했습니다.');await library(true)}
   catch(e){notice(e.message);purge.disabled=false;restore.disabled=false;purge.textContent='영구 삭제'}
  };
  actions.append(restore,purge);row.append(actions);$('trash-jobs').append(row)
 }
 if(!trash.length)$('trash-jobs').append(text('p','휴지통이 비어 있습니다.','hint'));
 }updateElapsedClocks();}catch(e){lastJobsKey=null;lastTrashKey=null;$('jobs').textContent=e.message}finally{libraryLoading=false;if(libraryRefreshRequested){libraryRefreshRequested=false;library()}}}
async function deleteLecture(id){try{
 await api(`/api/jobs/${id}`,editOptions('DELETE'));
 if(currentId===id){$('video').pause();$('video').removeAttribute('src');$('video').load();$('workspace').hidden=true;$('library').hidden=false;currentId=null;manifest=null;history.replaceState(null,'',location.pathname)}
 notice('강의를 삭제했습니다. 휴지통에서 복원할 수 있습니다.');await library(true);
 }catch(e){notice(e.message)}}
$('upload').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('button[type=submit]');b.disabled=true;showProcessingStatus({status:'running',message:'영상 업로드 중…',elapsed_seconds:0});try{const form=new FormData(e.target);if(!form.get('subtitles').size)form.delete('subtitles');const job=await api('/api/jobs',{method:'POST',headers:{'X-Lecture2Doc':'1'},body:form});showProcessingStatus({status:'queued',message:'처리 대기',elapsed_seconds:0});poll=setInterval(async()=>{try{const s=await api(`/api/jobs/${job.id}`);showProcessingStatus(s);await library(true);if(s.status==='complete'||s.status==='failed'){clearInterval(poll);b.disabled=false;if(s.status==='complete')await openJob(job.id)}}catch(err){clearInterval(poll);b.disabled=false;$('upload-status').textContent=err.message}},1800)}catch(err){$('upload-status').textContent=err.message;b.disabled=false}};
async function openJob(id){try{currentId=id;manifest=await api(`/api/jobs/${id}/manifest`);$('library').hidden=true;$('workspace').hidden=false;$('title').textContent=manifest.source.name;$('global-flags').textContent=manifest.quality_flags.filter(x=>!['fixed_camera_and_board_roi_require_review','sampling_can_miss_short_lived_writing','persistence_can_miss_short_lived_writing'].includes(x)).map(x=>labels[x]||x).join(' · ');$('video').src=file(manifest.playback);$('video-error').textContent='';$('source-download').href=file(manifest.source.path);$('manifest-download').href=`/api/jobs/${id}/manifest`;$('manifest-download').download='manifest.json';$('state-count').textContent=records().length;$('mode').querySelector('[value=restored]').textContent=manifest.board_pages?'B · 누적 판서 복원':'B · 기존 중간 상태 복원';$('mode').value=pageBAvailable()?'restored':'source';$('mode').querySelector('[value=restored]').disabled=!!manifest.board_pages&&!pageBAvailable();if(!manifest.board_pages)$('global-flags').textContent+=' · 이전 형식: 중간 관측 상태 목록입니다. 누적 판서는 재분석 후 표시됩니다.';$('global-flags').hidden=!$('global-flags').textContent;$('search').value='';selectedBoards.clear();renderDeletedBoards();renderStates();renderLearningAnalysis();selectState(0);history.replaceState(null,'',`#${id}`)}catch(e){$('upload-status').textContent=e.message}}
function updateBoardSelection(){
 $('selected-count').textContent=selectedBoards.size?`${selectedBoards.size}개 선택`:'';
 $('delete-selected-boards').disabled=selectedBoards.size===0;
 $('select-all-boards').disabled=records().length===0;
}
function renderLearningAnalysis(){
 const data=manifest?.learning_analysis;
 const pagesBox=$('learning-pages'),conceptsBox=$('learning-concepts');
 pagesBox.replaceChildren();conceptsBox.replaceChildren();
 
 if(!data){
  $('run-learning-analysis').textContent='분석 실행';
  $('learning-status').textContent='분석 결과 없음';
  $('learning-message').textContent='이 기록에는 학습 분석 결과가 없습니다. 새 브랜치의 코드로 강의를 다시 처리해 주세요.';
  pagesBox.append(text('p','표시할 중요도 분석이 없습니다.','hint'));
  conceptsBox.append(text('p','표시할 개념이 없습니다.','hint'));
  return;
 }
 $('run-learning-analysis').textContent='분석 다시 실행';
 const statusText={complete:'분석 완료',partial:'일부 자료 기반',insufficient_evidence:'근거 부족',failed:'분석 실패'};
 $('learning-status').textContent=statusText[data.status]||data.status||'분석 결과';
 const ai=data.ai_curation||{};
 $('learning-message').textContent=data.message||'반복·강조 표현과 필기·음성의 근거를 바탕으로 분석합니다.';
 if(!data.ai_curation){
  $('learning-status').textContent+=' · 새 분석 필요';
  $('learning-message').textContent='이전 분석 결과입니다. 아래의 분석 다시 실행을 눌러 AI 핵심 개념 선별을 적용해 주세요.';
 }
 if(ai.status==='complete') $('learning-status').textContent+=' · AI 개념 선별';
 else if(ai.status==='not_configured') $('learning-status').textContent+=' · 기본 필터';
 else if(ai.status==='failed') $('learning-status').textContent+=' · 기본 필터 대체';
 const visible=records();
 const ranked=(data.pages||[]).map(item=>({item,index:visible.findIndex(page=>page.id===item.board_page_id)}))
  .filter(row=>row.index>=0)
  .sort((a,b)=>(b.item.importance_score??-1)-(a.item.importance_score??-1));
 if(!ranked.length)pagesBox.append(text('p','중요도를 표시할 판서 구간이 없습니다.','hint'));
 for(const row of ranked.slice(0,8)){
  const item=row.item,page=visible[row.index],card=text('article','','learning-card');
  const head=text('div','','learning-card-heading');
  head.append(text('strong',`판서 ${boardNumber(page,row.index)}`),
   text('span',item.importance_score==null?'근거 부족':`중요도 ${item.importance_score}/100`,'importance-value'));
  const jump=text('button',`${fmt(item.start)} — ${fmt(item.end)} · 해당 판서 보기`,'learning-time-button');
  jump.onclick=()=>{selectState(row.index);seek(item.start)};
  card.append(head,jump);
  if(item.importance_score!=null){const meter=document.createElement('progress');meter.max=100;meter.value=item.importance_score;meter.setAttribute('aria-label','중요도');card.append(meter)}
  const reasons=text('p',(item.reasons||[]).slice(0,3).join(' · ')||'판단 근거가 충분하지 않습니다.','hint');
  card.append(reasons);
  const previewLabel=text('p','중요 구간 미리보기','learning-preview-label');
  const preview=document.createElement('video');
  preview.className='learning-preview-video';
  preview.controls=true;
  preview.playsInline=true;
  preview.preload='none';
  preview.setAttribute('aria-label',`판서 ${boardNumber(page,row.index)} 중요 구간 미리보기`);
  preview.dataset.start=String(Math.max(0,Number(item.start)||0));
  preview.dataset.end=String(Math.max(Number(item.start)||0,Number(item.end)||0));
  const posterFrame=manifest.source_frames.find(frame=>frame.id===page.selected_frame_id);
  if(posterFrame)preview.poster=file(posterFrame.image);
  preview.src=file(manifest.playback);
  preview.addEventListener('loadedmetadata',()=>{
   const startTime=Number(preview.dataset.start);
   const requestedEnd=Number(preview.dataset.end);
   const duration=Number.isFinite(preview.duration)?preview.duration:requestedEnd;
   const clipEnd=Math.min(requestedEnd,duration);
   preview.dataset.clipEnd=String(clipEnd);
   preview.currentTime=Math.min(startTime,Math.max(0,duration-0.1));
  },{once:true});
  preview.addEventListener('play',()=>{
   document.querySelectorAll('.learning-preview-video').forEach(other=>{
    if(other!==preview)other.pause();
   });
   $('video').pause();
  });
  preview.addEventListener('timeupdate',()=>{
   const clipEnd=Number(preview.dataset.clipEnd||preview.dataset.end);
   if(Number.isFinite(clipEnd)&&preview.currentTime>=clipEnd-0.08){
    preview.pause();
    preview.currentTime=Number(preview.dataset.start);
   }
  });
  card.append(previewLabel,preview);
  pagesBox.append(card);
 }
 const visibleIds=new Set(visible.map(page=>page.id));
 const concepts=(data.ai_curation?(data.concepts||[]):[]).filter(item=>
   !(item.board_page_ids||[]).length||(item.board_page_ids||[]).some(id=>visibleIds.has(id))
 ).slice(0,5);
 if(!data.ai_curation)conceptsBox.append(text('p','이전 결과의 개념 목록은 새 분석 전까지 숨겼습니다. 분석 다시 실행을 눌러 주제에 맞는 개념만 추려 주세요.','hint'));
 else if(!concepts.length)conceptsBox.append(text('p','주제와 연결되는 개념을 찾지 못했습니다. OCR 또는 음성 전사 자료를 확인해 주세요.','hint'));
 for(const concept of concepts){
  const card=text('article','','learning-card'),head=text('div','','learning-card-heading');
  head.append(text('strong',concept.name),
   text('span',concept.relevance_score==null?'관련도 미산정':`주제 관련도 ${concept.relevance_score}/100`,'importance-value'));
  card.append(head);
  if(concept.ai_reason)card.append(text('p',concept.ai_reason,'hint'));
  const appearances=[];
  for(const board of concept.board_appearances||[]){
   const idx=visible.findIndex(page=>page.id===board.board_page_id);
   if(idx>=0)appearances.push({kind:'board',start:board.start,end:board.end,index:idx,label:`필기 ${fmt(board.start)}`});
  }
  for(const speech of concept.speech_occurrences||[]){
   appearances.push({kind:'speech',start:speech.start,end:speech.end,label:`음성 ${fmt(speech.start)}`,excerpt:speech.text});
  }
  appearances.sort((a,b)=>a.start-b.start);
  if(!appearances.length)card.append(text('p','확인된 등장 시점이 없습니다.','hint'));
  for(const entry of appearances.slice(0,8)){
   const button=text('button',`${entry.label}${entry.end>entry.start?' — '+fmt(entry.end):''}`,'learning-time-button');
   if(entry.excerpt)button.append(text('small',entry.excerpt,'learning-excerpt'));
   button.onclick=()=>{
    if(entry.kind==='board'&&entry.index!==undefined)selectState(entry.index);
    seek(entry.start);
   };
   card.append(button);
  }
  conceptsBox.append(card);
 }
 
}
function renderStates(){const q=$('search').value.toLocaleLowerCase();$('states').replaceChildren();records().forEach((s,i)=>{
 const segs=manifest.transcript.segments.filter(t=>s.transcript_ids.includes(t.id));if(q&&!segs.some(t=>t.text.toLocaleLowerCase().includes(q)))return;
 const number=boardNumber(s,i),row=text('div','','state-item'),check=document.createElement('input');check.type='checkbox';check.checked=selectedBoards.has(s.id);check.setAttribute('aria-label',`판서 ${number} 선택`);check.onchange=()=>{if(check.checked)selectedBoards.add(s.id);else selectedBoards.delete(s.id);updateBoardSelection()};
 const b=text('button','',`state-button${i===currentState?' active':''}`),f=manifest.source_frames.find(f=>f.id===s.selected_frame_id),im=document.createElement('img');
 const rb=manifest.page_reconstructions?.find(r=>r.board_page_id===s.id);im.src=file($('mode').value==='restored'&&rb?rb.image:f.image);im.loading='lazy';im.alt=`판서 ${number}`;
 b.append(im,text('small',`${fmt(s.start)} — ${fmt(s.end)}`),text('span',`판서 ${number}`));b.onclick=()=>selectState(i);row.append(check,b);$('states').append(row)
 });updateBoardSelection()}
function renderDeletedBoards(){
 const deleted=manifest.deleted_boards||[];$('deleted-board-count').textContent=`(${deleted.length})`;$('deleted-boards').replaceChildren();
 for(const s of deleted){const row=text('div','','trash-row'),b=text('button','복원');row.append(text('span',`판서 ${s.display_number} · ${fmt(s.start)}`));b.onclick=()=>editBoards([s.id],false);row.append(b);$('deleted-boards').append(row)}
 if(!deleted.length)$('deleted-boards').append(text('p','삭제한 판서가 없습니다.','hint'));
}
async function editBoards(ids,deleting=true){
 if(!ids.length)return;
 const oldId=records()[currentState]?.id,oldIndex=currentState,job=currentId;
 try{await api(`/api/jobs/${job}/boards/${deleting?'delete':'restore'}`,editOptions('POST',{ids}));
  const updated=await api(`/api/jobs/${job}/manifest`);if(currentId!==job)return;manifest=updated;selectedBoards.clear();$('state-count').textContent=records().length;renderDeletedBoards();
  let next=records().findIndex(s=>s.id===oldId);if(next<0)next=Math.max(0,Math.min(oldIndex,records().length-1));selectState(next);
  notice(`${ids.length}개 판서를 ${deleting?'삭제':'복원'}했습니다.`);
 }catch(e){notice(e.message)}
}
function selectState(index){
 currentState=index;const s=records()[index];$('board-empty').hidden=!!s;$('board-content').hidden=!s;if(!s){$('evidence').replaceChildren();$('ocr-regions').replaceChildren();renderStates();renderTranscript();return;}
 const r=manifest.board_pages?manifest.page_reconstructions?.find(r=>r.board_page_id===s.id):manifest.reconstructed_boards.find(r=>r.board_state_id===s.id);
 const f=manifest.source_frames.find(f=>f.id===s.selected_frame_id);
 const restored=$('mode').value==='restored'&&!!r;
 $('board-title').textContent=`판서 ${boardNumber(s,index)}`;
 $('period').textContent=`${fmt(s.start)} — ${fmt(s.end)} · 대표 원본 ${fmt(f.time)}`;
 const p=restored?r.image:f.image;$('board-image').src=file(p);$('image-link').href=file(p);
 const best=s.selection_candidates?.[0];
 $('result-status').textContent=restored&&manifest.board_pages?`B · ${{composited:'원본 영역 합성',partial:'일부 가림 복원 · 미확인 영역은 A 원본 유지',original:['clear_original_sufficient','no_substantial_board_occluder','no_observed_hidden_writing'].includes(r.reason)?'A 원본 유지 · 추가 복원 불필요':r.reason==='clear_compatible_original_preferred'?'깨끗한 호환 원본 제공':'복원 보류 · A 원본 유지'}[r.status]}${r.coverage!=null&&r.status!=='original'?` · 감지 가림 영역 중 ${(r.coverage*100).toFixed(1)}% 대체`:''}`:manifest.board_pages?`A · 누적 판서의 선택 원본${best?` · 확인된 최종 판서 노출 ${(best.readable_fraction*100).toFixed(1)}% (추정)`:''}`:restored?`${{composited:'관측 영역 합성',partial:'부분 합성',fallback:'원본 제공'}[r.status]}`:'선택한 원본 프레임';
 $('quality').textContent=[...s.quality_flags,...(restored?r.quality_flags:[]),...(restored&&r.reason?[r.reason]:[])].map(x=>labels[x]||x).join(' · ')||'선택 원본';
 $('region-map').hidden=!restored||!r?.diagnostics?.length;if(restored&&r.diagnostics?.length)$('region-map').href=file(r.diagnostics[0]);
 $('provenance').hidden=!restored||!r?.provenance;if(restored&&r.provenance)$('provenance').href=file(r.provenance);
 $('evidence-count').textContent=`(전체 관측 ${s.source_frame_ids.length}개)`;
 const drawEvidence=all=>{
  $('evidence').replaceChildren();
  let ids=s.source_frame_ids;
  if(!all&&ids.length>32){const chosen=new Set([f.id,...(restored?r.source_frame_ids:[]),...(restored?r.verification_frame_ids||[]:[]),...(s.selection_candidates||[]).map(c=>c.source_frame_id)]);for(let j=0;j<24;j++)chosen.add(ids[Math.round(j*(ids.length-1)/23)]);ids=ids.filter(id=>chosen.has(id));}
  for(const fid of ids){const sf=manifest.source_frames.find(f=>f.id===fid);const b=text('button','');const im=document.createElement('img');im.src=file(sf.image);im.loading='lazy';im.alt=`원본 ${fmt(sf.time)}`;const candidate=s.selection_candidates?.find(c=>c.source_frame_id===fid);b.append(im,text('small',`${fmt(sf.time)}${fid===f.id?' · A 선택 원본':''}${restored&&r.source_frame_ids.includes(fid)?' · B 사용':''}${restored&&r.verification_frame_ids?.includes(fid)?' · B 대조 관측':''}${candidate?.readable_fraction!==undefined?` · 노출 ${(candidate.readable_fraction*100).toFixed(1)}%`:''}`));b.onclick=()=>seek(sf.time);$('evidence').append(b)}
  if(!all&&ids.length<s.source_frame_ids.length){const show=text('button',`모든 원본 관측 ${s.source_frame_ids.length}개 보기`);show.onclick=()=>drawEvidence(true);$('evidence').append(show)}
 };drawEvidence(false);renderStates();renderTranscript();renderOCR();
}
function renderTranscript(){const s=records()[currentState],all=$('all-transcript').checked;const tr=manifest.transcript;$('transcript-status').textContent=`${tr.status}${tr.language?' · '+tr.language:''}${tr.message?' · '+tr.message:''}`;$('transcript').replaceChildren();const segments=tr.segments.filter(t=>all||s?.transcript_ids.includes(t.id));if(!segments.length)$('transcript').append(text('p','표시할 전사문이 없습니다.','hint'));segments.forEach(t=>{const b=text('button','','cue');b.dataset.start=t.start;b.dataset.end=t.end;b.append(text('time',`${fmt(t.start)} — ${fmt(t.end)}`),document.createTextNode(t.text));b.onclick=()=>seek(t.start);$('transcript').append(b)})}
function seek(t){$('video').currentTime=t;$('video').play().catch(()=>{})}
$('video').ontimeupdate=()=>{const t=$('video').currentTime;document.querySelectorAll('.cue').forEach(e=>e.classList.toggle('playing',t>=Number(e.dataset.start)&&t<Number(e.dataset.end)))};
$('video').onerror=()=>{$('video-error').textContent='이 브라우저에서 재생할 수 없는 코덱입니다. 등록 시 H.264/AAC 사본 생성을 선택하거나 원본을 외부 플레이어에서 여세요.'};
$('mode').onchange=()=>selectState(currentState);$('search').oninput=renderStates;$('all-transcript').onchange=renderTranscript;$('back').onclick=()=>{$('workspace').hidden=true;$('library').hidden=false;$('video').pause();history.replaceState(null,'',location.pathname);library()};
$('run-learning-analysis').onclick=async()=>{
 const job=currentId,button=$('run-learning-analysis');if(!job)return;
 button.disabled=true;button.textContent='분석 중…';
 try{
  await api(`/api/jobs/${job}/learning-analysis`,editOptions('POST'));
  if(currentId!==job)return;
  manifest=await api(`/api/jobs/${job}/manifest`);
  renderLearningAnalysis();
  selectState(Math.min(currentState,Math.max(0,records().length-1)));
  notice('강의 중요도와 개념 등장 시간 분석을 업데이트했습니다.');
 }catch(e){notice(e.message)}
 finally{button.disabled=false;button.textContent=manifest?.learning_analysis?'분석 다시 실행':'분석 실행'}
};
library();if(/^[a-f0-9]{24}$/.test(location.hash.slice(1)))openJob(location.hash.slice(1));

function renderOCR(){
 const page=records()[currentState],ocr=manifest.ocr;
 const result=ocr?.pages?.find(p=>p.board_page_id===page.id);
 $('ocr-regions').replaceChildren();$('qwen-review-result').replaceChildren();$('ocr-actions').hidden=true;$('ocr-message').textContent='';
 $('qwen-review').disabled=!result;
 if(!result){$('ocr-status').textContent=ocr?.status==='failed'?'처리 실패':'미실행';$('ocr-message').textContent=ocr?.message||'이 기록에는 OCR 결과가 없습니다.';return}
 const recognizer=(result.engine||{}).models?.rec;
 $('ocr-status').textContent=`${result.regions.length}개 영역 · ${ocr.language==='ko'?'한글·영문':'영문'}${recognizer?' · '+recognizer:''}`;
 $('ocr-message').textContent=`${result.input.output} 이미지에서 인식 · 원본 ${fmt(result.input.time)}`;
 $('ocr-actions').hidden=false;$('ocr-text').href=file(result.text_file);$('ocr-json').href=file(result.json_file);$('ocr-overlay').href=file(result.overlay);
 $('ocr-copy').onclick=async()=>{try{await navigator.clipboard.writeText(result.text);$('ocr-message').textContent='텍스트를 복사했습니다.'}catch(e){$('ocr-message').textContent='TXT 파일에서 텍스트를 복사할 수 있습니다.'}};
 if(!result.regions.length)$('ocr-regions').append(text('p','인식된 글자가 없습니다.','hint'));
 for(const region of result.regions){
  const row=text('div','','ocr-region');const a=document.createElement('a');a.href=file(region.image);a.target='_blank';a.rel='noopener';
  const im=document.createElement('img');im.src=file(region.image);im.loading='lazy';im.alt=`인식 영역 ${region.order}`;a.append(im);
  const body=text('div','');body.append(text('p',region.text),text('small',`${region.order} · 인식 점수 ${(region.score*100).toFixed(0)}%${region.review?' · 낮은 점수':''}`));
  row.append(a,body);$('ocr-regions').append(row);
 }
 const qwen=result.qwen_review;
 if(qwen){
  const box=text('section','','qwen-review-box');
  const heading=text('div','','qwen-review-heading');
  heading.append(text('h3','Qwen 이미지 재검토 결과'),text('small',`${qwen.model||'Qwen'} · ${Number(qwen.elapsed_seconds||0).toFixed(1)}초 · 원본 OCR 유지`));
  box.append(heading);
  if(!(qwen.blocks||[]).length&&qwen.text)box.append(text('pre',qwen.text,'qwen-review-text'));
  for(const block of qwen.blocks||[]){
   const row=text('div','','qwen-review-block');
   row.append(text('span',block.kind==='formula'?'수식':block.kind==='title'?'제목':block.kind==='list'?'목록':'문장','qwen-block-kind'));
   row.append(text('p',block.text||''));
   if(block.latex)row.append(text('code',block.latex,'qwen-latex'));
   const refs=(block.source_ocr_regions||[]).join(', ');
   row.append(text('small',`판독 ${block.confidence||'미표시'}${refs?' · 기존 OCR 영역 '+refs:''}`));
   box.append(row);
  }
  if((qwen.unreadable_regions||[]).length){
   const unreadable=text('div','','qwen-unreadable');
   unreadable.append(text('strong','판독이 어려운 부분'));
   for(const item of qwen.unreadable_regions)unreadable.append(text('p',item));
   box.append(unreadable);
  }
  $('qwen-review-result').append(box);
 }
}

$('ocr-rerun').onclick=async()=>{
 const job=currentId,button=$('ocr-rerun');if(!job)return;
 button.disabled=true;button.textContent='OCR 재인식 중…';
 $('ocr-message').textContent='새 OCR 모델로 기존 판서 이미지를 다시 읽고 있습니다. A/B 판서와 원본 영상은 변경하지 않습니다.';
 try{
  const result=await api(`/api/jobs/${job}/ocr`,editOptions('POST'));
  if(currentId!==job)return;
  manifest=await api(`/api/jobs/${job}/manifest`);
  renderOCR();
  $('ocr-message').textContent=result.message+` · 판서 ${result.pages}장 · ${Number(result.elapsed_seconds||0).toFixed(1)}초`;
 }catch(e){
  $('ocr-message').textContent=e.message;
 }finally{
  button.disabled=false;button.textContent='OCR 다시 실행';
 }
};
$('qwen-review').onclick=async()=>{
 const job=currentId,page=records()[currentState],button=$('qwen-review');if(!job||!page)return;
 button.disabled=true;button.textContent='Qwen 분석 중…';
 $('ocr-message').textContent='선택 판서 이미지와 기존 OCR 후보를 로컬 Qwen 모델에 전달하고 있습니다. 외부 API는 사용하지 않습니다.';
 try{
  const result=await api(`/api/jobs/${job}/qwen-review`,editOptions('POST',{board_page_id:page.id}));
  if(currentId!==job)return;
  manifest=await api(`/api/jobs/${job}/manifest`);
  renderOCR();
  $('ocr-message').textContent=result.message+` · ${result.blocks}개 인식 블록 · ${Number(result.elapsed_seconds||0).toFixed(1)}초`;
 }catch(e){
  $('ocr-message').textContent=e.message;
 }finally{
  button.disabled=false;button.textContent='Qwen 판서 재검토';
 }
};
$('delete-lecture').onclick=()=>deleteLecture(currentId);
$('delete-current-board').onclick=()=>{const s=records()[currentState];if(s)editBoards([s.id])};
$('delete-selected-boards').onclick=()=>editBoards([...selectedBoards]);
$('select-all-boards').onclick=()=>{const all=selectedBoards.size===records().length;selectedBoards.clear();if(!all)records().forEach(s=>selectedBoards.add(s.id));renderStates()};
