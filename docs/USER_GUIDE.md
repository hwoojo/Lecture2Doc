# Lecture2Doc · V1

이 문서는 이전 루트 README의 사용법·기능 설명을 옮겨 보존한 상세 안내입니다. 새 팀원용 개요와 저장소 구조는 [루트 README](../README.md)를 먼저 참고하세요.

모델 가중치는 저장소에 포함하지 않습니다. 필요한 모델을 명령으로 내려받으면 저장소의 `models/` 폴더에 저장되고, 이후에는 로컬에서 사용합니다.

각 누적 판서의 삭제·교체 전 원본을 고르는 A와, 실제 관측 영역을 합성하는 B를 구현했습니다. 최신 AccessMath 결과는 **Lecture 01 4장 / Lecture 02 16장 / Lecture 06 part 1 6장**입니다. 기존 계산 계약으로 A/B F1은 각각 **96.75 / 99.90 / 97.54%**이며, 정상 실행 26장은 원본을 유지했습니다. 별도 실제 가림 구간 8건 중 7건에서 필기 추가 노출을 확인했지만, 부분 인물·패치 경계가 남습니다. **C는 판서 OCR을 구현했습니다.** [V1 OCR 실행 결과](V1_OCR.md)를 함께 확인하세요. [최신 원인 분석·비교·한계](AB_REFINEMENT.md), [이미지·후보·출처 비교](../experiments/ab-refinement/review/index.html)를 확인하세요. 이전 [A 보고서](BOARD_PAGES.md), [B 보고서](PAGE_RESTORATION.md), [세부 상태 진단](QUALITY_IMPROVEMENT.md)과 실험 파일은 기준선으로 보존합니다.

강의 영상의 판서와 발언을 **원본으로 확인할 수 있게 기록하는 로컬 웹 앱**입니다. 기본 A/B/C 처리에는 외부 계정, API 키, CUDA, LLM/VLM이 필요하지 않습니다. Qwen 기반 판서 재검토와 AI 개념 선별은 Ollama와 로컬 Qwen 모델을 설치한 경우 선택적으로 실행됩니다.

**V1은 A(누적 판서 선택) + B(관측 기반 복원) + C(OCR)**입니다. C는 판서의 문자를 읽어 텍스트·영역 좌표·원본 crop을 저장하며 웹에서 확인·복사할 수 있습니다. 수식→LaTeX는 후속 목표로 남깁니다. 전사 보정·강의 노트·편집·통합 검색은 D로 분리합니다. [단계별 범위](ROADMAP.md), [V1 OCR 실행 보고서](V1_OCR.md)를 확인하세요.

## 실행 파일

프로젝트의 **`serve.sh` 하나만 실행**하면 됩니다. 실행 위치를 자동으로 맞추고 브라우저에서 `http://127.0.0.1:8765`를 엽니다.

```bash
./serve.sh
```

실행 중 Python 코드 변경은 자동 반영합니다. 화면 파일은 내용 해시를 붙인 주소로 제공해 구버전 브라우저 캐시와 분리하며, 새로고침하면 갱신됩니다. 이미 같은 앱이 실행 중이면 기존 화면을 엽니다. 종료는 실행한 터미널에서 `Ctrl+C`입니다. 다른 포트는 `./serve.sh --port 8888`, 브라우저 자동 열기 생략은 `./serve.sh --no-browser`로 지정합니다. 기존 `data/`의 내 강의를 그대로 사용하며 시연용 강의를 자동 추가하지 않습니다.

## 시작

Python 3.11 이상과 `ffmpeg`/`ffprobe`가 필요합니다. H.264 사본 생성에는 설치한 FFmpeg에 `libx264`, AAC 인코더가 있어야 합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ocr]'
lecture2doc ocr-models --language en
lecture2doc serve
```

[http://127.0.0.1:8765](http://127.0.0.1:8765)에서 영상을 등록합니다. 기존 SRT/VTT/JSON 자막을 넣거나 분석 설정에서 Whisper 모델을 지정할 수 있습니다. 한글·영문이 섞인 판서의 기본 OCR은 PP-OCRv5 Korean Mobile입니다. `lecture2doc ocr-models --language ko`를 실행하면 기본 모델과 선택 실험용 한국어 손글씨 모델을 함께 준비합니다. 손글씨 모델은 `LECTURE2DOC_KO_RECOGNIZER=handwriting`을 설정한 경우에만 선택됩니다. 음성 전사가 필요하면 `python -m pip install -e '.[stt]'`를 추가로 실행한 뒤 첫 다운로드를 허용합니다. 모델 파일은 각자 저장소의 `models/`에 쌓이며 이후에는 네트워크 없이 사용합니다. 로컬 CTranslate2 모델 폴더도 지정할 수 있습니다.

Whisper 모델은 첫 실행에서 내려받을 수 있습니다. 예를 들어 `lecture2doc process lecture.mp4 --out data/my-lecture --model base --language ko --allow-download`는 모델을 `models/whisper/base/`에 저장합니다. 웹 화면에서도 전사 설정에서 모델을 지정하고 첫 다운로드를 허용할 수 있습니다. 다른 저장 위치는 `LECTURE2DOC_WHISPER_MODELS`로 바꿀 수 있습니다.

CLI로 만든 기록을 웹에서 열려면 기록 폴더를 지정합니다:

```bash
lecture2doc serve --record data/my-lecture --port 8886
```

웹에서 `기록 열기`를 누르면 누적 판서와 선정 근거 원본을 확인할 수 있습니다. CLI 분석 결과도 `serve --record 결과폴더`로 엽니다. `--data`는 새 웹 업로드가 저장되는 폴더입니다.

MP4V·MKV 등 브라우저 재생 비호환 형식은 자동으로 H.264/AAC 사본을 생성합니다. `--make-playback`은 입력 형식과 관계없이 변환을 강제합니다. 변환 실패 시 원본을 남기고 경고를 표시합니다.

## CLI

```bash
# 자막을 사용한 CPU 누적 판서 기록
lecture2doc process lecture.mp4 --out data/my-lecture \
  --subtitles lecture.srt --board blackboard --roi 0.05,0.05,0.9,0.8

# CPU int8 음성 전사: 최초 공개 모델 다운로드만 네트워크 사용
lecture2doc process lecture.mp4 --out data/my-lecture \
  --model small --language ko --allow-download

# 미리 준비한 모델을 사용한 오프라인 실행
lecture2doc process lecture.mp4 --out data/my-lecture \
  --model /absolute/path/ctranslate2-whisper-small --language ko

# 원본 키프레임만 제공 / 브라우저용 H.264·AAC 사본
lecture2doc process lecture.mkv --out data/my-lecture \
  --no-reconstruct --make-playback

# 지정 단계만 새 캐시 폴더에서 다시 실행; 기존 근거 파일은 유지
lecture2doc process lecture.mp4 --out data/my-lecture \
  --rerun pages --rerun page-restoration

lecture2doc serve --record data/my-lecture
```

### C OCR

```bash
# 기존 A/B를 보존하고 OCR 결과를 새 기록에 추가
lecture2doc ocr experiments/ab-refinement/accepted/lecture01 \
  --out experiments/v1/new-lecture01 --language en

# 한글·영문 OCR을 포함한 새 영상 처리
lecture2doc process lecture.mp4 --out data/my-v1 --ocr ko

# 기존 V1의 OCR 재실행 / 전처리 비교
lecture2doc ocr-models --language ko
lecture2doc ocr data/my-v1 --language ko --rerun
lecture2doc ocr data/my-v1 --language ko --preprocessing adaptive
```

각 실행의 전체 시간과 단계별 캐시 사용 여부는 `manifest.json`의 `elapsed_seconds`, `timings`에 기록합니다. 기존 기록에 OCR만 추가하면 이번 시간은 `ocr.elapsed_seconds`로 따로 남깁니다. [실제 실행 시간](V1_OCR.md)을 참고하세요.

OCR은 `.[ocr]` 선택 설치와 `lecture2doc ocr-models --language en` 또는 `--language ko` 모델 다운로드가 필요합니다. OCR 모델 선택을 바꾼 뒤에는 기존 강의의 C · 판서 텍스트에서 **OCR 다시 실행**을 누르거나 CLI에서 `--rerun`으로 다시 생성해야 합니다. 결과 상태 줄에 실제 인식기 이름이 표시됩니다. 기본 모델 위치는 `models/ocr`, 다른 위치는 `LECTURE2DOC_OCR_MODELS`로 지정합니다. CLI 새 영상 처리는 기본 영문 OCR이며 `--ocr off`로 생략합니다. 웹 등록에서 언어를 선택할 수 있습니다. OCR 실패 시 A/B 결과와 오류 메시지를 보존합니다.

### 전체 필수 모델 한 번에 준비

OCR과 Qwen 모델을 한 번에 내려받으려면 Ollama 앱을 설치하고 실행한 뒤 저장소 루트에서 다음 명령을 실행합니다.

```bash
lecture2doc setup-models
```

이 명령은 영문·한글 OCR 모델을 모두 준비한 다음 Mac 메모리에 맞는 Qwen 비전 모델을 내려받습니다. 기본 Qwen 자동 선택은 32GB 이상 Apple Silicon에서 최신 Qwen3.8-27B MLX, 24GB 이상에서 8B, 16GB 이상 24GB 미만에서 4B, 16GB 미만에서 2B를 사용합니다. Ollama 버전이나 연결 상태를 먼저 검사하므로 조건을 만족하지 않으면 모델 다운로드를 시작하지 않습니다.

기존처럼 모델을 개별 설치할 수도 있습니다. OCR은 `lecture2doc ocr-models --language ko`, Qwen은 `lecture2doc qwen-models`를 사용합니다.

### 로컬 Qwen Vision 설치와 사용

Qwen으로 판서 이미지를 재검토하고 핵심 개념을 선별하려면 Ollama 앱을 설치하고 실행합니다. `lecture2doc qwen-models`는 Mac 메모리를 감지해 최신 Qwen3.8-27B MLX 또는 기기 메모리에 따라 Qwen3-VL 8B/4B/2B를 자동 선택합니다. Qwen3.8-27B는 약 18GB 다운로드와 Ollama 0.40.0 이상이 필요합니다.

```bash
lecture2doc qwen-models
```

기존 강의에서 C · 판서 텍스트의 **Qwen 판서 재검토**를 누르면 선택된 판서 이미지와 OCR 후보를 로컬 모델이 다시 읽습니다. Qwen 결과는 기존 OCR을 덮어쓰지 않고 별도 영역에 저장합니다. D · 학습 분석의 **분석 다시 실행**도 동일한 Qwen 모델을 사용해 주제 핵심 개념을 최대 5개 선별합니다.

모델 목록은 `lecture2doc qwen-models --list`, 설치 여부는 `lecture2doc qwen-models --check`로 확인합니다. 최신 모델을 강제하려면 `lecture2doc qwen-models --profile latest`, 8B 균형형은 `--profile balanced`, 4B 저메모리 모델은 `--profile compact`, 2B 초저메모리 모델은 `--profile ultra-compact`를 사용합니다. 모델을 바꾼 뒤에는 서버 실행 전에 `LECTURE2DOC_QWEN_MODEL` 환경 변수를 지정하고 해당 모델을 내려받습니다. Qwen 추론은 로컬 Ollama에서만 실행되므로 API 키나 외부 AI 토큰 비용이 필요하지 않습니다. 최초 다운로드에는 인터넷이 필요합니다. [자세한 안내](QWEN_LOCAL.md)

B 결과가 있으면 B를, 없으면 A를 읽습니다. 텍스트를 눌러 고치는 기능·통합 검색은 D에서 다룹니다. OCR에 수식이 문자열로 나타날 수 있지만 LaTeX로 변환하지 않습니다.

`--roi x,y,w,h`는 영상 크기 대비 0~1 좌표입니다. 실제 칠판 영역만 지정하는 것이 권장됩니다. 자동 판별은 배경 표면과 필기 극성을 사용합니다. A의 테두리 추정은 긴 테두리 쌍이 확인되는 경우에 한하며 일반적인 칠판 위치 검출 기능은 아닙니다. 초기 기본 영역은 전체 영상입니다.

기본 분석 간격은 0.5초입니다. 이는 후보를 관측하는 간격이며, 최종 키프레임은 판서 변화에 따라 선택됩니다. `--sample-seconds 0.25` 등으로 줄일 수 있습니다. 간격보다 짧은 필기, 너무 작거나 희미한 필기는 누락될 수 있습니다. 모든 관측 원본은 보존합니다.

## 제공 기능

- 영상의 실제 PTS 기반 시간과 원본 해상도 PNG 보존. 가변 프레임률 테스트 포함.
- 판서 추가·삭제/수정의 지속 관측으로 상태 분리. 이전 상태를 버리지 않아 삭제 전 판서를 보존.
- 넓은 화면 변화와 카메라 이동에서 구간 분리. 불확실한 상태에서는 복원 대신 원본 제공.
- 움직이는 큰 전경을 색·두께로 추정하고, **같은 상태의 다른 시점에서 반복 관측된 영역**만 대체하는 부분 복원.
- 복원 이미지와 원본 분리 저장. 타일별 원본 프레임 ID 및 이미지 좌표를 JSON으로 제공.
- faster-whisper CPU/int8, 모델·언어 선택, SRT/VTT/JSON 입력, 전사 원문 보존 및 캐싱.
- 로컬 웹 업로드, 백그라운드 처리 상태, 판서별 전사문, 전체 전사, 검색, 영상 시점 이동, 원본 목록, 품질 경고.
- 디코딩·분석·음성 추출·전사·복원·재생 사본 단계 캐시. 실패한 전사는 성공한 영상 분석을 지우지 않음. 분석 실패 시 추출 원본과 자막을 보존.

누적 판서 B 상태는 `composited`(감지된 칠판 가림 영역을 거의 모두 대체), `partial`(미확인 영역에 원본 유지), `original`(원본 제공)입니다. 이전 세부 상태 B에는 `fallback`도 남아 있습니다. 합성에 사용한 원본과 대체 영역을 함께 표시합니다.

## 처리 상태

강의 등록 후 현재 단계와 **경과 시간**을 표시합니다. 경과 시간은 1초마다 갱신되며, 단계 메시지가 오랫동안 바뀌지 않아도 계속 증가합니다. 대기 시간은 처리 시간과 구분합니다. 페이지를 새로 열어도 내 강의 목록에서 진행 중인 작업과 경과 시간을 확인할 수 있습니다. 완료되면 최종 소요 시간으로 고정됩니다.

## 강의·판서 삭제

- **내 강의 → 삭제** 또는 강의 화면의 **강의 삭제**로 목록에서 제거합니다. 내 강의의 **휴지통 → 복원**으로 되돌립니다.
- 판서 썸네일의 체크박스에서 여러 장을 고른 뒤 **선택 삭제**를 누릅니다. 현재 판서만 지우려면 **이 판서 삭제**를 누릅니다.
- **삭제한 판서 → 복원**으로 되돌릴 수 있습니다. 삭제 상태는 새로고침·서버 재시작 뒤에도 유지합니다. 남은 판서 번호는 원래 번호를 유지합니다.
- 휴지통의 **영구 삭제**는 확인 후 업로드한 강의 폴더와 원본 영상·A/B/C·전사 파일을 실제로 삭제합니다. 공유 `models/`와 다른 강의는 유지합니다. `--record`로 연 외부 기록은 목록 등록만 영구 삭제하고 외부 폴더는 유지합니다.
- 일반 삭제는 복원할 수 있고 영구 삭제는 복원할 수 없습니다. 처리 중인 강의는 완료 후 삭제할 수 있습니다.

## 결과 구조

```text
결과폴더/
  source.mp4                 # 입력 사본 (확장자는 입력에 따름)
  source.json                # 입력 이름, SHA-256
  manifest.json              # 현재 실행 결과, 스키마 버전 1
  cache/
    decode/<key>/            # SourceFrame PNG + stage.json
    analysis/<key>/          # BoardState, SelectedFrame 근거, 측정값
    audio/<key>/             # 16 kHz 모노 WAV
    transcript/<key>/        # TranscriptSegment 원문·시각
    pages/<key>/             # 누적 판서·후보 점수·말기 필기/빈 영역 근거
    page-restoration/<key>/  # 누적 판서 B·영역별 원본/대조 관측·밝기 계수
    ocr/<key>/               # C 텍스트·영역·원본 crop·모델·입력 해시
    reconstruction/<key>/    # 이전 세부 상태 B 진단
    playback/<key>/          # 선택적 H.264/AAC 사본
```

입력 SHA-256·설정·알고리즘 버전으로 캐시를 분리합니다. 자막 내용이나 로컬 모델 파일이 바뀌면 전사 캐시가 달라집니다. `--rerun`은 이전 캐시를 삭제하지 않습니다. 다른 영상을 같은 결과 폴더에 넣으면 거부합니다. 디스크 절약을 위한 자동 원본 삭제 기능은 없습니다. 기본 0.5초 PNG 추출은 장시간 1080p 영상에서 많은 저장 공간과 디코딩 시간이 필요합니다.

### 기본·실험 OCR 모델 선택

기본값은 한국어와 영어 혼합 텍스트용 PP-OCRv5 Korean Mobile입니다. 손글씨 특화 모델을 비교하려면 서버 실행 전에 다음을 설정합니다.

```bash
export LECTURE2DOC_KO_RECOGNIZER=handwriting
```

기본 모델로 돌아가려면 다음을 실행합니다.

```bash
unset LECTURE2DOC_KO_RECOGNIZER
```

설정 변경 후 서버를 재시작하고 OCR을 다시 실행해야 합니다.

### 기존 강의의 OCR만 다시 실행

1. `lecture2doc ocr-models --language ko`로 새 모델을 준비합니다.
2. 서버를 종료하고 다시 실행합니다.
3. 기존 강의의 **C · 판서 텍스트**에서 **OCR 다시 실행**을 누릅니다.
4. 기본 모델을 사용했다면 상태에 `korean-PP-OCRv5-mobile`이 표시되는지 확인합니다. 실험 손글씨 모델을 선택한 경우에만 `korean-handwriting-rec`이 표시됩니다.

이 기능은 저장된 A/B 판서 이미지를 다시 읽습니다. 영상 프레임 추출·A/B 복원·음성 전사는 다시 실행하지 않습니다. 이전 OCR 파일은 캐시에 남고 새 결과만 manifest에 연결됩니다.

## 검증과 재현

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/evaluate.py --root experiments/new-run
```

동일 입력과 설정으로 다시 실행하면 단계 캐시를 재사용합니다. 합성 생성기 코드를 수정한 경우 새 실험 폴더를 지정하세요. 기존 다른 입력을 덮어쓰지 않습니다.

고정 버전 설치(선택 STT·테스트 포함)는 `python -m pip install -r requirements-lock.txt` 후 `python -m pip install --no-deps -e .`로 가능합니다. 저장소는 Linux x86_64 / Python 3.11에서 개발·검증했습니다.

- [실험 비교 HTML](../experiments/validated/index.html): 원본 관측 / 선택 키프레임 / 복원 / 정답 병렬 비교
- [검증 결과와 실패 사례](VALIDATION.md)
- [AccessMath 데이터셋 실행·정답 비교](ACCESSMATH.md)
- [참고 저장소 코드 조사](RESEARCH.md)
- [아키텍처와 데이터 계약](ARCHITECTURE.md)
- [실제 영상 재현 절차](../scripts/real_example.sh)

![웹 뷰어](viewer-desktop.png)

## 알려진 한계

전경 마스크는 사람 분할 모델이 아닙니다. 칠판과 비슷한 옷, 정지한 사람, 넓은 그림·굵은 필기, 반사광, 저대비 필기, 조명 변화, 복잡한 배경에서 실패할 수 있습니다. 색이 다양한 판서를 일반적으로 검증하지 않았습니다. 1 분석 픽셀 이내의 변화는 코덱 경계 흔들림으로 간주하므로 아주 작은 수정은 감지하지 못할 수 있습니다.

정합은 고정 카메라 가정을 검사하는 수준입니다. 이동·줌을 보정하는 파노라마 복원, 정확한 장면 전환 인식, 교수자 추적은 없습니다. 컷이 색 변화 없이 일어나면 별도 장면 ID가 생기지 않을 수 있으며, 내용 불일치 검사가 복원을 제한합니다.

실제 MIT 예제는 480×360이고 카메라 이동이 있어 초기 지원 조건 밖입니다. 45초에서 46개 상태로 과분할됐고 복원 46건 모두 원본으로 대체됐습니다. 추가한 AccessMath 1080p 실행에서는 공식 학습 분할의 첫 영상 파일 전체와 정답 키프레임 3개를 비교했으며, 기본 A/B 회수율은 모두 96.42%로 복원 개선이 없었습니다. **전체 데이터셋 테스트 분할, 강의 전체의 판서 누락률, 한국어 STT 정확도는 아직 검증하지 못했습니다.** Whisper tiny 영어 실행 성공을 전사 정확성 평가로 간주하지 않습니다.

단일 사용자 localhost용입니다. 서버를 재시작하면 대기 작업을 자동 재개하지 않으며, 같은 출력 폴더의 CLI 재실행으로 캐시를 재사용합니다. 작업 취소·서버 다중 프로세스·사용자 인증·PDF·LaTeX·노트 생성은 구현 범위에 없습니다.
