# V1 — A・B・C OCR

2026-11-02. **A＝누적 판서 선택, B＝관측 기반 복원, C＝문자 OCR**로 묶어 V1으로 명명했다. 패키지 버전은 `1.0.0`이다. 수식→LaTeX는 후속 목표로 옮겼다.

## 구현한 기능

- CPU 로컬 OCR: RapidOCR 3.10.0 / ONNX Runtime. 검출은 PP-OCRv6 small, 영문 인식은 PP-OCRv6 medium, 한글·영문 인식은 PP-OCRv5 Korean Mobile을 기본으로 사용.
- B 이미지가 있으면 B를 읽고, 없으면 A를 읽는다. 분석용 축소 마스크가 아닌 원본 해상도 이미지를 사용한다.
- 사용자 ROI를 적용하고 결과 좌표는 전체 이미지의 원본 픽셀 좌표로 돌려 저장한다.
- 문장별 텍스트·모델 점수·polygon/bbox·원본 crop과 판서 ID, 입력 이미지 해시, 근거 프레임 ID·시각, B 영역별 출처 JSON을 연결한다.
- 웹의 **C · 판서 텍스트**에서 원본 영역과 인식 결과를 나란히 표시한다. 텍스트 복사, TXT·JSON, 전체 인식 영역 지도를 제공한다.
- 새 영상 처리에 C를 연결하고, 기존 A/B 기록에도 OCR만 추가할 수 있다. 언어·모델 해시·전처리·입력 이미지에 따라 캐시를 분리한다. `--rerun`은 이전 캐시를 남긴다.
- 모델 준비 명령과 인식을 분리했다. 준비 후 인식은 로컬에서 실행하며 외부 API를 호출하지 않는다.
- 영상 파이프라인에서 OCR이 실패해도 A/B 결과는 남기고 C 오류를 기록한다.

## 실제 영상 실행

기존 `experiments/ab-refinement/accepted/`는 유지하고 `experiments/v1/`에 별도 기록을 만들었다.

| 입력 | 누적 판서 | OCR 영역 | 기록된 OCR 시간 |
|---|---:|---:|---:|
| AccessMath Lecture 01 | 4 | 52 | 24.63초 |
| AccessMath Lecture 02 testing | 16 | 230 | 101.16초 |
| Lecture 01 적응 이진화 비교 | 4 | 50 | 17.14초 |

CPU 4스레드 설정으로 실행했다. 시간은 페이지별 처리 시간의 합이며 모델 생성·파일 복사는 제외한다. 일부 비교 실행은 동시에 진행했다.

**A/B manifest 필드와 OCR 입력 이미지 해시는 기준선과 일치한다.** 이번 작업에서 선택·복원 알고리즘은 바꾸지 않았다. [검증 기록](../experiments/v1/verification.json).

## 한 영상의 처리 시간

**Core Ultra 7 268V에서 17분 26초 AccessMath Lecture 01을 처음부터 처리하는 데 12분 31초가 걸렸다.** 새 결과 폴더에서 프레임 추출·A/B·C 캐시 없이 실행했다. 1080p 원본, 0.5초 관측 간격, 최종 판서 4장, OCR 52개 영역이다. OCR은 ONNX Runtime CPU 4스레드를 사용했다. 음성 전사와 최초 모델 다운로드는 제외했다.

| 단계 | 소요 시간 |
|---|---:|
| 원본 해상도 프레임 2,090장 추출·저장 | 3분 13.4초 |
| 판서 변화 분석 | 3분 40.2초 |
| 누적 판서·대표 원본 선택 | 2분 51.8초 |
| 내부 세부 상태 180개 진단용 복원 | 2분 19.5초 |
| 최종 누적 판서 B | 0.4초 |
| C OCR — 모델 생성·4장 인식 포함 | **25.1초** |
| 전체 — 입력 확인·결과 저장 포함 | **12분 31.4초** |
| 동일 입력·설정의 캐시 재사용 실행 | **1.3초** |

첫 실행 시간의 대부분은 프레임 추출·A/B 분석이다. 진단용 복원도 현재 기본 실행 경로에 포함된다. 최종 출력은 누적 판서 4장이다. 캐시 실행은 재분석 없이 기존 결과를 읽은 시간이다.

43분 20초 Lecture 02는 기존 A/B 판서 16장에 OCR을 추가했으며 페이지 인식 시간 합계가 **1분 41초**였다. 이 영상의 처음부터 전체 처리 시간은 이번에 측정하지 않았다.

계측 자료: [처음 실행](../experiments/v1/timing/cold.json), [캐시 실행](../experiments/v1/timing/warm.json), [단계 진행 로그](../experiments/v1/timing/run.log), [재현 스크립트](../experiments/v1/timing/run.py). 처음 실행과 캐시 실행 manifest를 각각 보존했다. 중간에 서버 재시작으로 끊긴 실행은 별도로 보존하고 위 시간에서 제외했다.

전체 시간에 OCR이 빠지던 기록을 수정했다. 새 실행은 `elapsed_seconds`에 OCR까지 포함하고, `timings`에 단계별 시간과 캐시 적중 여부를 저장한다. 관련 회귀 테스트 4개가 통과했다. [시간 기록 테스트](../experiments/v1/timing/tests.txt).

## 사람이 읽은 문장과 비교

Lecture 01·02의 네 판서에서 문장·라벨 **26곳**을 골라 원본을 읽고 전사를 작성했다. 검출된 영역 25곳과 미검출 문장 1곳이다. 실행 후 오류를 확인하기 위해 고른 진단 표본이며, 아래 점수는 이 표본에 대한 결과다. 영어 대소문자와 구두점을 유지하고 Unicode NFC 및 공백 정리만 적용했다. 정답은 OCR 입력이나 선택에 사용하지 않았다.

| 진단 표본 | 영역 | 글자 오류율 CER ↓ | 단어 오류율 WER ↓ | 전체 일치 |
|---|---:|---:|---:|---:|
| Lecture 01 | 13 | **34.58%** | 57.58% | 8/13 |
| Lecture 02 | 13 | **4.33%** | 25.71% | 11/13 |
| 합계 | 26 | **19.67%** | 41.18% | 19/26 |

[원본 영역 / 사람 전사 / OCR 비교 HTML](../experiments/v1/review/index.html), [좌표·정답·출력·계산 JSON](../experiments/v1/ocr-audit.json), [계산 재현](../experiments/v1/evaluate_ocr.py).

잘 읽힌 예시는 `Consistent system`, `Multiplication`, `PARALLEL lines`다. 실패 사례는 다음과 같다.

- `What is represented by this equation ?`를 `Wh?`로 읽었다. 영역은 잡았지만 긴 손글씨의 인식이 실패했다.
- `two lines ⇒ two linear equations` 문장을 누락했다.
- `System of linear Equations`의 공백·앞 번호가 빠졌다. 글자가 대부분 맞아도 단어 오류율이 커진다.
- `System`을 `Sys am`, `3rd-dim`을 `3-din`으로 읽었다. 높은 모델 점수에서도 오자가 나왔다.

적응 이진화는 첫 판서의 긴 질문을 놓치고 다른 문구의 오류도 늘렸다. **RGB를 기본으로 유지**하고 이진화는 CLI 비교 옵션으로 제공한다. 현재 OCR의 핵심 개선점은 긴 흘림체 문장과 줄 누락이다.

이전 한글 모델은 다운로드·해시 확인 후 실행해 6개 영역 출력을 확인했다. 초기 손글씨 인식기 기본 적용은 실제 칠판 영상에서 영어 혼합 판서가 악화되는 문제가 확인되어 기본값에서 제외했다. 현재 기본값은 한국어·영어를 지원하는 PP-OCRv5 Korean Mobile이고, `korean-handwriting-rec`는 환경 변수로 선택하는 실험 모델이다. PP-OCRv6 영역 탐지와 원본 좌표·crop 저장 구조는 유지한다.

## 2026-11-10 한국어 손글씨 인식기 비교 모델 추가 및 기본값 복구

- 한국어 손글씨 모델 `korean-handwriting-rec`를 비교용으로 추가하고 모델 출처·Apache-2.0 라이선스·파일 해시를 `models/ocr/ko.json`에 기록하도록 했다.
- 기본 인식기는 PP-OCRv5 Korean Mobile로 복구했으며, 실험 모델은 `LECTURE2DOC_KO_RECOGNIZER=handwriting`으로만 선택한다.
- OCR 알고리즘 버전을 올려 모델 선택이 달라지면 이전 인식 결과 캐시를 재사용하지 않는다.
- 모델은 종이 손글씨 데이터에서 높은 결과를 보고했지만 칠판·분필·수식에 대한 검증은 아직 수행하지 않았다. [모델 평가와 PaddleOCR-VL 검토](OCR_MODEL_EVALUATION.md)를 참고한다.
- `tests/test_ocr.py`에 기본 모델 다운로드, 실험 모델 다운로드, 기본값·선택값 전환을 확인하는 테스트를 추가했다. 이 대화에서 테스트를 실제 실행하지는 않았으므로 Mac에서 `python -m pytest tests/test_ocr.py -q`로 확인해야 한다.

## 자동 검증과 연결 확인

- **52개 테스트 통과**: 기존 A/B 회귀와 OCR의 B 입력 선택, ROI 좌표 복원, 원본 crop, 캐시/강제 실행, 기준선 보존, 웹 산출 파일 허용·비공개 파일 차단. [테스트 로그](../experiments/v1/tests.txt).
- 마지막 CLI·웹 처리 상태 변경 후 관련 테스트 **5개 재통과**. [추가 로그](../experiments/v1/tests-web-final.txt).
- 합성 `long_hold` 영상도 **영상 입력 → A → B → C → manifest**까지 실행했다. C complete, 판서 1장. 이는 기능 연결 확인용 실행이다.
- 실제 웹에서 판서 1→4 전환 시 C 텍스트와 crop·TXT/JSON 링크가 함께 바뀌는 것을 확인했다.

![V1 OCR 화면](../experiments/v1/viewer-ocr.jpg)

## 재현

```bash
python -m pip install -e '.[ocr]'
lecture2doc ocr-models --language en
lecture2doc ocr-models --language ko

lecture2doc ocr experiments/ab-refinement/accepted/lecture01 \
  --out experiments/v1/new-lecture01 --language en
lecture2doc serve --record experiments/v1/new-lecture01 --port 8886
```

모델 URL·SHA-256은 `models/ocr/en.json`, `ko.json`에 있고 각 OCR 결과 JSON에도 복사된다. 추가 의존성 버전은 `requirements-ocr-lock.txt`에 기록했다. OCR 자체에는 음성 전사나 API 키가 필요하지 않다.

## 남은 작업

V1의 A/B/C 실행과 결과 확인 기능을 연결했다. OCR 품질 개선은 긴 문장 인식·줄 누락·읽기 순서, 실제 한글 필기 평가 순으로 진행한다. 다른 인식기와 이미지 입력 API는 같은 입력으로 비교할 수 있다. 수식→LaTeX는 후속 목표이며, 전사 보정·강의 노트·편집·통합 검색은 D에 둔다.
