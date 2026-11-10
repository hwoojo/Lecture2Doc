# 로컬 Qwen Vision 설치 및 사용

## 목적

Lecture2Doc은 Ollama를 통해 Qwen 비전·언어 모델을 Mac 안에서 실행한다. 같은 모델이 다음 기능을 모두 담당한다.

- 판서 이미지와 기존 OCR 후보 비교
- 한글·영문·수식의 문맥 기반 재판독
- 불확실한 글자와 수식 영역 표시
- 음성 전사와 판서 후보를 함께 읽고 핵심 개념 최대 5개 선별

추론은 `127.0.0.1`의 Ollama 서비스에서 실행한다. OpenAI 등 외부 AI API와 토큰 요금은 사용하지 않는다. 모델을 처음 내려받을 때만 인터넷 연결이 필요하며, 다운로드된 가중치는 Ollama가 별도 모델 저장소에 관리한다. 모델 파일 자체는 GitHub에 포함하지 않는다.

## 최신 모델과 자동 선택

2026년 11월 기준 Qwen 최신 공개 로컬 비전·언어 모델은 Qwen3.8-27B다. 공식 모델 가중치는 Apache-2.0으로 공개되어 있고 이미지 입력을 지원한다. Ollama에는 Apple Silicon용 `qwen3.8:27b-mlx` 태그가 있으며 다운로드 크기는 약 18GB다.

공식 자료:
- [Qwen3.8 공식 저장소](https://github.com/QwenLM/Qwen3.8)
- [Qwen3.8-27B 모델 카드](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Ollama Qwen3.8 모델](https://ollama.com/library/qwen3.8)
- [Ollama Qwen3.8 태그 목록](https://ollama.com/library/qwen3.8/tags)

모델은 Mac 통합 메모리에 따라 자동 선택된다.

| 기기 메모리 | 자동 선택 모델 | 다운로드 크기 | 안내 |
|---|---|---:|---|
| Apple Silicon 32GB 이상 | `qwen3.8:27b-mlx` | 약 18GB | 최신 27B 모델 |
| 24GB 이상 | `qwen3-vl:8b` | 약 6.2GB | 균형형 |
| 16GB 이상 24GB 미만 | `qwen3-vl:4b` | 약 3.3GB | 저메모리 호환 |
| 16GB 미만 | `qwen3-vl:2b` | 약 1.9GB | 초저메모리 호환 |
| 메모리 확인 불가 | `qwen3-vl:4b` | 약 3.3GB | 안전한 기본값 |

27B 모델은 약 18GB를 내려받고 실행 시 모델 가중치와 컨텍스트 메모리도 필요하다. 24GB Mac에서는 여유가 부족할 수 있어 자동 선택은 32GB부터 권장한다. 48GB 이상이면 더 여유롭게 사용할 수 있다. 메모리 자동 감지가 실패하면 작은 4B 모델을 선택한다.

다른 모델을 쓰려면 `LECTURE2DOC_QWEN_MODEL` 환경 변수로 고정할 수 있다.

## 설치 및 필수 모델 다운로드

1. [Ollama for macOS](https://ollama.com/download)를 설치하고 Ollama 앱을 실행한다.
2. 터미널에서 버전을 확인한다.

```bash
ollama --version
```

3. Lecture2Doc 저장소에서 가상환경을 활성화하고 Qwen 모델을 준비한다.

```bash
cd ~/Lecture2Doc
source .venv/bin/activate
lecture2doc setup-models
```

통합 명령은 영문·한글 OCR 모델과 메모리에 맞는 Qwen 모델을 한 번에 준비한다. Qwen만 별도로 설치하려면 `lecture2doc qwen-models`를 사용한다. 기본 Qwen 명령은 메모리를 감지해 권장 모델을 고르고, 필요한 모델을 Ollama 저장소로 내려받는다. 27B Qwen3.8 모델을 선택한 경우 Ollama 0.40.0 이상이 필요하다. Qwen3-VL 4B/8B 모델은 Ollama 0.12.7 이상이 필요하다.

지원 모델과 설치 크기를 확인하려면 다음을 실행한다.

```bash
lecture2doc qwen-models --list
```

선택된 모델이 이미 설치되어 있는지 내려받기 없이 확인하려면 다음을 실행한다.

```bash
lecture2doc qwen-models --check
```

프로필을 직접 고를 수도 있다.

```bash
# 자동 선택: 기본값
lecture2doc qwen-models --profile auto

# 최신 27B 모델을 강제로 선택 (Apple Silicon 32GB 이상 권장)
lecture2doc qwen-models --profile latest

# 8B 균형형
lecture2doc qwen-models --profile balanced

# 4B 저메모리 모델
lecture2doc qwen-models --profile compact

# 2B 초저메모리 모델
lecture2doc qwen-models --profile ultra-compact

# 모델을 직접 지정
lecture2doc qwen-models --model qwen3.8:27b-mlx
```

최신 27B 모델을 강제로 선택하면 메모리 부족 기기에서도 다운로드할 수 있으므로, 기기 사양을 먼저 확인한다.

## 기존 강의에서 판서 재검토

1. `lecture2doc qwen-models`로 모델을 준비한다.
2. Lecture2Doc 서버를 실행한다.
3. 기존 강의를 열고 원하는 판서 페이지를 선택한다.
4. C · 판서 텍스트의 **Qwen 판서 재검토**를 누른다.

Qwen은 선택된 판서 원본 이미지와 기존 OCR 후보를 함께 읽는다. 결과는 기존 OCR을 덮어쓰지 않고 별도 `qwen_review` 필드로 저장한다.

- 판서의 읽기 순서를 유지한 텍스트 블록
- 한국어·영어·수식 유형 구분
- 수식의 LaTeX 후보
- 시각적 판독 신뢰도(높음·보통·낮음)
- 판독하기 어려운 영역과 비고

## 강의 핵심 개념 선별

D · 학습 분석은 동일한 Qwen 모델을 텍스트 모드로 사용한다. 판서 OCR 후보와 음성 전사 일부, 반복 횟수와 원본 시간 정보를 입력해 강의 주제를 파악하고 핵심 개념을 최대 5개 선별한다.

- 일반 표현·말버릇을 제외한다.
- 입력 후보에 없는 개념을 새로 만들지 않는다.
- 기존 음성 등장 시각과 판서 페이지 연결을 보존한다.
- 중요도 점수는 기존 근거 기반 규칙으로 계산한다.

학습 분석의 **분석 다시 실행** 버튼을 누르면 Qwen 개념 선별도 다시 수행한다. 모델이 없거나 Ollama에 연결할 수 없으면 프로그램은 기본 필터로 대체한다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LECTURE2DOC_QWEN_MODEL` | 메모리별 자동 선택 | 사용할 Ollama 모델 이름 |
| `LECTURE2DOC_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 로컬 API 주소 |

최신 Mac 모델을 고정하려면:

```bash
export LECTURE2DOC_QWEN_MODEL=qwen3.8:27b-mlx
```

자동 선택으로 되돌리려면:

```bash
unset LECTURE2DOC_QWEN_MODEL
```

모델 다운로드가 성공하면 선택된 모델명이 저장소의 `models/qwen/selection.json`에 기록되어 다음 실행에도 유지된다. 이 파일은 Git에서 제외된다. 환경 변수 `LECTURE2DOC_QWEN_MODEL`은 저장된 선택보다 우선한다. 모델을 새로 자동 추천하려면 `lecture2doc setup-models`를 다시 실행한다. 환경 변수를 바꾼 뒤에는 Lecture2Doc 서버를 재시작한다. 이미 설치된 모델을 다른 태그로 선택한 경우 해당 모델도 `lecture2doc qwen-models --model <모델명>`으로 내려받아야 한다.

## 로컬 실행과 네트워크

- 이미지와 전사문을 처리하는 추론 요청은 로컬 Ollama로만 전달한다.
- OpenAI 등 외부 AI API로 텍스트나 이미지가 전송되지 않는다.
- 최초 모델 다운로드에는 인터넷이 필요하다.
- 모델이 설치된 후에는 Ollama 로컬 서버와 Lecture2Doc을 오프라인으로 사용할 수 있다.
- 로컬 추론은 Mac의 통합 메모리·GPU/CPU·전력을 사용하며 모델 크기와 이미지 해상도에 따라 처리 시간이 달라진다.

## 오류 해결

- **Ollama 명령을 찾을 수 없음:** Ollama macOS 앱을 설치하고 실행한 뒤 새 터미널에서 확인한다.
- **Ollama 연결 실패:** 앱이 실행 중인지 확인하고 `http://127.0.0.1:11434/api/version`을 연다.
- **Ollama 0.40.0 미만:** Qwen3.8-27B는 최신 Ollama로 업데이트한다. 작은 Qwen3-VL 모델은 0.12.7 이상을 지원한다.
- **모델을 찾을 수 없음:** `lecture2doc qwen-models`를 실행한다.
- **메모리 부족 또는 처리 지연:** `lecture2doc qwen-models --profile compact`로 4B 모델을 내려받는다. 메모리가 16GB보다 적다면 `lecture2doc qwen-models --profile ultra-compact`로 2B 모델을 내려받고, `unset LECTURE2DOC_QWEN_MODEL`로 환경 변수를 해제한 뒤 서버를 재시작한다.
- **Qwen 결과가 틀림:** Qwen 결과는 보조 판독이다. 원본 이미지와 기존 OCR을 함께 비교하고 읽을 수 없는 영역은 추측하지 않는다.
