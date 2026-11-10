# 구조와 데이터 계약

## 기술 선택

Python 한 프로세스 + 단일 백그라운드 작업 큐 + FastAPI + 외부 CDN 없는 HTML/CSS/JavaScript를 사용합니다. React/Node 빌드, DB, 클라우드 계정은 필요하지 않습니다. JSON과 콘텐츠 주소 기반 디렉터리로 원본과 파생물을 구분합니다. 영상은 PyAV로 순차 디코딩하고 FFmpeg는 음성 추출 및 선택적 재생 사본을 담당합니다. OpenCV가 색·모폴로지·연결 요소·phase correlation을 처리합니다. faster-whisper는 선택 설치입니다.

PySceneDetect는 조사했지만 핵심 의존성으로 넣지 않았습니다. 일반 장면 검출이 판서 상태와 같지 않고, 최소 구성에서 이미 넓은 화면 변화와 이동 제한 검사가 필요하기 때문입니다. 현재 장면 검출은 완전하지 않습니다. 동적 카메라 보정은 지원하지 않습니다.

## 처리

1. 입력 SHA-256, 크기·길이·오디오 정보를 확인하고 원본을 복사합니다.
2. PyAV PTS에서 영상 시작 시간을 빼 관측 시각을 저장합니다. 원본 크기 PNG를 보존합니다. 명목 FPS로 시간을 재계산하지 않습니다.
3. 지정 ROI를 최대 가로 640픽셀로 축소합니다. 강건한 타일 색·밝기 적합으로 매끄러운 표면 조명을 측정하고 두꺼운 전경과 연결된 피부 영역을 가림 후보로 봅니다. 가림을 제외한 얇은 필기 극성으로 칠판 유형을 판단합니다. 측정된 표면 이미지는 분석용이며 출력에 사용하지 않습니다.
4. 내부 `board_states`는 관측 변화 진단을 보존합니다. 사용자 출력 `board_pages`는 추가 필기를 누적하고 지속적으로 보이는 삭제·수정에서 완성 판서를 보존합니다. 작은 국소 수정, 최근 작성 활동, 삭제 중간 장면의 잡음을 구분합니다. 억제한 삭제 장면은 이전 판서의 근거로 귀속합니다.
5. 카메라 이동은 공동 가시 배경 특징점/ECC로 확인합니다. 작은 검증된 이동은 분석 마스크만 정렬하고, 말기 근거는 최종 선택 원본 좌표로 돌려 내보냅니다. 불확실한 B 정렬 후보는 제외하며 공간 워핑한 픽셀을 합성하지 않습니다.
6. A는 말기 필기 노출·필기 가림·다른 내용의 비율로 원본 후보를 비교합니다. 5 분석 픽셀 이내 내용/점수 동등 후보는 전경이 적은 원본을 우선합니다. 모든 관측과 후보 점수, 말기 필기 마스크, 칠판 마스크, 마지막 빈 영역/필기 관측 시각을 보존합니다. GT는 읽지 않습니다.
7. 누적 판서 B는 실제 칠판 내부의 전경 중심이 확인된 경우에만 시도합니다. 같은 판서의 삭제 경계·정렬·영역별 말기 내용·빈 영역의 시각을 확인하고 1초 이상 떨어진 두 관측에서 글자/색이 일치하는 영역을 복사합니다. 깨끗한 원본 하나가 충분하면 우선 사용합니다. 타일을 세분할 수 있고, 공동 가시 빈 칠판으로 검증한 밝기 보정만 선택적으로 적용합니다. 미관측 영역은 A 원본으로 유지하고 잔상·필기 보존 검증 실패 시 합성을 취소합니다. 이전 세부 상태 B 결과는 별도 진단으로 유지합니다.
8. 전사는 별도 캐시로 수행하며 원문을 저장합니다. 구간 겹침으로 연결하고 의미 관계를 추정하지 않습니다.
9. C OCR은 준비된 로컬 ONNX 모델로 B(없으면 A) 이미지를 읽습니다. 문장·영역·원본 crop·입력 해시와 모델 정보를 별도 OCR 캐시에 저장합니다.
10. D 학습 분석은 강조·반복·교차 근거로 판서 구간 중요도를 추정합니다. Ollama의 로컬 Qwen 비전·언어 모델이 설치되어 있으면 전사·OCR 텍스트 후보에서 강의 주제와 핵심 개념을 최대 5개 선별합니다. 모델 다운로드 명령은 Mac 메모리에 따라 최신 Qwen3.8-27B MLX 또는 Qwen3-VL 8B/4B를 자동 선택합니다. 각 개념의 음성 등장 시각과 판서 페이지 시간 범위는 별도 캐시에 저장하며, Ollama가 없으면 로컬 기본 필터를 사용합니다.
11. C의 선택 판서 재검토는 동일한 Ollama Qwen 비전·언어 모델에 판서 이미지와 기존 OCR 후보를 전달합니다. 결과는 OCR 페이지의 `qwen_review` 필드에 별도로 저장하며 기존 OCR 문자열·좌표·crop은 덮어쓰지 않습니다.
11. 현재 manifest를 원자적으로 기록합니다. 재처리로 바뀌는 manifest와 달리 이전 원본·복원 캐시 파일은 남습니다.

가려진 동안 실제로 지워지고 다시 쓰인 내용을 정확히 추적할 수는 없습니다. 관측에 없는 사실은 복원하지 않습니다. 상태 경계와 마스크는 추정이며 원본 대조가 필요합니다.

## manifest.json v1

| 개념 | 필드 | 의미 |
|---|---|---|
| SourceFrame | `source_frames[]` | id, time(초), pts, time_base, decode_index, width, height, image |
| BoardPage | `board_pages[]` | 누적 완성 판서, 시작/끝, 대표/동등 후보, 원본 목록, 말기 필기·칠판·시간별 부재 근거, 품질 표시 |
| PageReconstruction | `page_reconstructions[]` | 누적 판서 B의 original/partial/composited, 사유, 원본/대조 관측, 영역 지도, 출처 |
| BoardState | `board_states[]` | id, scene, start/end, source_frame_ids, selected_frame_id, boundary_reason, quality_flags, transcript_ids |
| SelectedFrame | `selected_frames[]` | board_state_id, source_frame_id, reason |
| ReconstructedBoard | `reconstructed_boards[]` | board_state_id, status, image, source_frame_ids, provenance, coverage, quality_flags |
| TranscriptSegment | `transcript.segments[]` | id, start/end(초), text(원문) |
| QualityFlag | 전체·상태·복원 `quality_flags[]` | 기계 판독 가능한 문자열. UI에는 한국어 설명 또는 원문 코드 표시 |
| 측정 디버그 | `analysis_metrics[]` | 시각, ink_pixels, occlusion, sharpness, additions/removals, cut, shift, event |
| OCR | `ocr.pages[]` | 판서별 텍스트·인식 영역·원본 crop·입력과 모델 정보·산출 파일 |
| LearningAnalysis | `learning_analysis` | 페이지 중요도·판단 신호·AI 선별 개념·개념별 음성 및 판서 등장 시간·AI 선별 상태 |
| 제품 이름 | `product_version` | A+B+C를 묶은 V1 |
| 처리 시간 | `elapsed_seconds`, `elapsed_scope`, `timings` | 영상 입력부터 OCR까지의 전체 시간, 포함 범위, 단계별 초·캐시 적중 여부 |

구간은 `[start, end)` 의미입니다. 원본 발언은 전체 전사 목록에 한 번 저장하고 겹치는 상태는 ID로 참조하므로 전체 전사에서 유실되지 않습니다. 상태 화면에서는 경계를 걸치는 발언이 여러 상태에 보일 수 있습니다. `temporal_links_only=true`는 시간 기반 연결임을 명시합니다.

누적 판서 B의 `provenance` 좌표는 전체 원본 프레임 픽셀입니다. 각 사각형은 x/y/width/height/source_frame_id/witness_frame_id/source_box와 선택적 photometric_correction을 가집니다. 밝기 계수와 ROI 좌표로 PNG를 재생할 수 있습니다. 복원 출력은 전체 프레임이며 가림 영역 밖은 A와 같습니다. `coverage`는 감지된 칠판 가림 영역 중 대체한 비율로, 글자 회수율이나 교수자 전체 제거율이 아닙니다. `composited`도 영상 하단의 몸통이 없다는 뜻은 아닙니다. 이전 `reconstructed_boards`의 ROI 좌표·verified 타일 계약과 구분합니다.

V1의 C는 OCR만 담당합니다. `ocr.pages[]`에 `board_page_id`, 입력 이미지·해시·A/B 구분·근거 프레임·B provenance, 원본 해상도 polygon/bbox, text·score·원본 crop을 저장합니다. raw JSON·TXT·영역 지도·영역별 PNG를 별도 캐시에 보존하고 `artifacts` 목록으로 웹 제공 파일을 제한합니다. `score`는 인식기 출력 점수이며 정답 확률로 보정한 값은 아닙니다. 모델 파일·설정·언어·전처리·입력 이미지가 캐시 키에 포함됩니다. OCR 예외는 영상 파이프라인에서 A/B 보존 후 `ocr.status=failed`로 기록합니다.

`product_version=V1`은 제품 이름이며 `schema_version=1`의 저장 형식과 구분합니다. 음성 전사는 C의 필수 입력이 아닙니다. 수식→LaTeX는 후속 목표, 전사 보정·노트·편집·통합 검색은 D입니다. 최신 범위는 [단계별 개발 범위](ROADMAP.md)를 따릅니다.

전체 실행의 `elapsed_seconds`는 OCR까지 포함하고 마지막 manifest 저장은 제외합니다. 기존 결과에 OCR만 추가한 경우 원래 전체 실행 시간을 유지하고, `ocr.elapsed_seconds`에 이번 OCR 호출 시간(모델 생성·캐시 확인 포함), `ocr.annotation_elapsed_seconds`에 기록 복사 시간을 포함한 추가 작업 시간을 남깁니다. 각 `ocr.pages[].elapsed_seconds`는 해당 페이지를 처음 인식한 시간이며 캐시 재사용 시간과 구분합니다. 예전 OCR 실행 자료는 페이지별 시간만 기록되어 있습니다.

## 실패·캐시·운영

디코딩·분석·오디오·전사·복원·재생은 독립 캐시를 가집니다. 캐시 키에는 앱 알고리즘 버전이 포함됩니다. 분석 설정 변경으로 전사를 다시 하지 않으며, 자막 변경으로 영상 디코딩을 반복하지 않습니다. 전사 실패는 캐시 성공으로 기록하지 않습니다. `--rerun`은 새 디렉터리에 실행합니다. 재실행 manifest만 현재 결과로 교체되고 과거 근거는 유지됩니다. 강제 재실행 뒤 일반 재실행에서는 기존 기본 키 캐시를 다시 쓸 수 있습니다.

분석 예외 시 관측 프레임별 보수적 상태를 만들어 복원을 제한합니다. 디코딩 예외 시 전체 시각 결과는 비어 있을 수 있지만 자막/음성 단계는 실행합니다. 입력 메타데이터를 전혀 읽을 수 없거나 저장 공간 부족이면 처리가 중단될 수 있습니다. OS 프로세스 강제 종료 시 단계 파일은 남지만 완료 manifest가 없을 수 있습니다.

원본 PNG 디스크 사용량은 강의 길이에 비례합니다. 영상 프레임 전체를 RAM에 올리지 않으며 누적 판서 B는 근접/분산 후보와 필기 관측 시각의 후보를 조사하고 원본 이미지 캐시를 제한합니다. 후보 수는 고정 48개가 아닙니다. 현 구현은 PNG 인코딩·IO 비용이 크므로 전체 강의 처리 속도를 보증하지 않습니다.

서버는 localhost에만 바인딩하고 허용 Host를 제한합니다. 변경 요청은 같은 출처와 커스텀 헤더를 요구합니다. 파일 제공은 manifest가 참조한 파일만 허용하고 디렉터리 탈출을 거부합니다. 전사문과 파일명은 DOM textContent로 표시합니다. 브라우저 재생은 HTTP Range 요청을 지원합니다. 이 제한은 다중 사용자 접근 제어의 대체가 아닙니다.

## 사용자 삭제 기록

강의 삭제는 데이터 루트의 `library-edits.json`, 판서 삭제는 각 기록의 `user-edits.json`에 저장한다. 원본 manifest와 캐시는 변경하지 않는다. 삭제된 강의는 일반 목록·manifest·파일 API에서 제외하고 휴지통에서 복원한다. `--record`로 연 외부 기록은 절대 경로를 식별자로 사용해 다른 실험의 동일 가상 ID와 구분한다.

판서 삭제는 원본 영상 해시·판서 ID·구간·선택 프레임에 연결한다. 재분석으로 내용이 달라진 같은 ID를 자동으로 숨기지 않는다. 웹 manifest는 삭제된 판서와 연결된 B/OCR 결과를 제외하고 복원 가능한 `deleted_boards` 목록을 제공한다. 원래 순서의 `display_number`를 유지한다. 변경 요청의 기존 동일 출처 검사와 헤더 검사는 삭제·복원에도 적용한다. 실행 중인 작업의 편집은 409로 거부한다.

## 처리 경과와 영구 삭제

웹 작업의 `status.json`에는 `queued_at`, `started_at`, `updated_at`, `finished_at`, `elapsed_seconds`를 남긴다. 실행 중 GET 응답은 monotonic clock으로 현재 경과 시간을 계산하므로 단계 보고가 없는 동안에도 시간이 증가한다. 브라우저는 받은 시간을 기준으로 1초마다 표시하고 상태를 조회한다. 목록 내용이 같으면 버튼 DOM을 유지하고 타이머만 갱신한다. 완료·실패·중단 상태는 계속 증가시키지 않는다.

`POST /api/jobs/{id}/purge`는 휴지통 항목과 `confirmed=true`가 필요하다. 관리하는 업로드 폴더만 실제 삭제하며 경로 이탈·폴더 심볼릭 링크·실행 중 작업은 거부한다. 외부 `--record`는 등록을 제거하고 `purged_imports`에 남겨 재시작 후 자동으로 다시 나타나지 않게 한다. 외부 원본과 공유 모델은 삭제하지 않는다. UI는 삭제 범위를 보여주는 확인창을 거친다.
