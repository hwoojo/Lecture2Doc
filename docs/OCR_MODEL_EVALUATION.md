# OCR 모델 검토 및 한국어 손글씨 모델 교체

작성일: 2026년 11월 10일

## 결론

- 한국어·영어 혼합 판서의 기본 인식기는 PP-OCRv5 Korean Mobile로 복구했다. `korean-handwriting-rec`는 선택 실험 모델로 유지한다.
- 글자 영역 탐지기는 PP-OCRv6 small을 유지했다. 새 모델은 글자 인식기(recognizer)이며, 글자 영역을 찾는 detector를 대체하지 않는다.
- PaddleOCR-VL 1.6은 Apache-2.0으로 공개된 오픈소스 모델이다. 다만 Lecture2Doc의 핵심 OCR을 바로 대체하기보다 수식·복잡한 판서 페이지를 재검사하는 보조 엔진으로 검증하는 것이 적절하다.

## 1. 한국어 손글씨 인식기

모델: [Rebeccaaaaaaaa/korean-handwriting-rec](https://huggingface.co/Rebeccaaaaaaaa/korean-handwriting-rec)

코드: [VerifiedIdiot/korean-handwriting-rec](https://github.com/VerifiedIdiot/korean-handwriting-rec)

라이선스: Apache-2.0 (모델·코드)

### 모델 특징

- PP-OCRv5 한국어 모바일 인식 모델을 기반으로 손글씨와 인쇄 공문서 데이터로 추가 학습한 ONNX 인식기다.
- 약 15MB로 안내되어 있으며 CPU에서 실행할 수 있고 외부 API가 필요 없다.
- PP-OCRv5와 입출력 계약 및 11,945자 사전을 유지하므로 RapidOCR 3.x에서 인식 모델 경로와 사전 경로를 바꾸는 방식으로 연결할 수 있다.
- 공식 모델 카드가 자체 보고한 평가: 손글씨 단어 정확도 97.4%(검증 단어 2만 개), 학습에 없던 필자 15명의 손글씨 99.1%, 흘림체 문장 CER 7.8%. 해당 수치는 모델 제작자가 공개한 데이터셋 평가이며, Lecture2Doc의 칠판 영상에 대한 독립 평가 결과가 아니다.

### Lecture2Doc 반영

손글씨 모델을 기본값으로 적용한 뒤 실제 강의 사진에서 영문 혼합 인식이 악화되는 것이 확인되어, 기본 모델을 한국어·영어를 지원하는 PP-OCRv5 Korean Mobile로 되돌렸다. 손글씨 모델은 비교 실험용으로 보존한다.

기본 파이프라인은 다음과 같다.

```text
원본 또는 B 복원 이미지
  → PP-OCRv6 small: 글자 영역 탐지
  → PP-OCRv5 Korean Mobile: 한국어·영어 글자 인식
  → PP-OCRv5 사전
  → 기존 OCR 영역·좌표·원본 crop 저장
```

바뀐 파일은 `lecture2doc/ocr.py`다. 한국어 모델 준비 명령은 기존과 같다.

```bash
lecture2doc ocr-models --language ko
```

이 명령은 `models/ocr/ko.json`을 기본·실험 인식 모델의 파일명·SHA-256·출처 정보로 갱신한다. 이전 PP-OCRv5 인식기 파일은 자동 삭제하지 않는다. 인식 단계는 계속 로컬 ONNX Runtime CPU로 실행하며 새 필수 패키지를 추가하지 않았다.

모델과 사전은 `models/ocr/`에 저장된다. 기본 PP-OCRv5 Korean Mobile의 인식 모델·사전과 실험용 `rec_korean_hw.onnx`, `ppocrv5_korean_dict.txt`, 모델 인덱스 `ko.json`을 함께 보관한다.

기본 모델과 실험 모델을 같은 `ko.json` 인덱스에 함께 기록한다. 기본은 PP-OCRv5 Korean Mobile이며, `LECTURE2DOC_KO_RECOGNIZER=handwriting`을 설정한 경우에만 손글씨 실험 모델을 선택한다. OCR 알고리즘 버전을 올려 모델 선택이 바뀌면 이전 인식 결과 캐시를 재사용하지 않는다.

### 적용 후 기존 강의 OCR 갱신

모델 변경만으로 기존 강의의 OCR 결과가 저절로 다시 만들어지지는 않는다. 서버를 재시작한 뒤 해당 강의의 **C · 판서 텍스트** 영역에서 **OCR 다시 실행**을 누르면 기존 A/B 판서 이미지를 새 모델로 다시 읽는다. 완료 후 C 영역에 실제 인식기 이름이 표시된다. 기본 모델은 `korean-PP-OCRv5-mobile`, 선택 실험 모델은 `korean-handwriting-rec`이다. 이 버튼은 영상 디코딩·A/B 복원은 다시 하지 않는다.

터미널에서 실행하고 싶은 경우 기존 강의별로 다음 명령을 사용할 수 있다.

```bash
lecture2doc ocr data/<강의ID> --language ko --rerun
```

`--rerun`은 이전 OCR 파일과 기록을 보존하고 새 OCR 결과를 별도 캐시로 만든다. 실제 입력 확장자와 OCR 출력 위치는 원래 기록 설정을 따른다.

### 적용 한계

손글씨 실험 모델의 높은 공개 수치는 종이 문서의 단어·문장 손글씨 평가다. 칠판·분필·화이트보드·낮은 대비·장거리 촬영·수학 기호 인식이 같은 수준으로 개선된다고 보장할 수 없다. 특히 글자 검출 단계가 한 줄을 너무 작게 나누거나 긴 문장을 놓치면, 인식기만 교체해서는 해결되지 않는다. 수식의 구조적 해석과 LaTeX 변환도 제공하지 않는다.

따라서 현재는 기본값이 아니며, 실제 강의에서 기준 모델과 CER/WER를 비교한 뒤 선택 여부를 판단해야 한다.

## 2. PaddleOCR-VL 1.6 오픈소스 여부

공식 모델: [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)

공식 코드: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

공식 안내: [PaddleOCR-VL 1.6 소개](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html)

라이선스: 모델 카드의 LICENSE 표시 기준 Apache-2.0. 공식 PaddleOCR 저장소도 Apache-2.0이다.

모델 카드의 전체 저장소는 약 1.93GB이고, `model.safetensors`가 약 1.92GB다. 0.9B 파라미터 VLM을 중심으로 하며, 전체 문서 파싱에서는 레이아웃 분석과 VLM 인식을 순서대로 사용한다.

### 장점

- 한글을 포함한 다국어 문서 인식
- 복잡한 문서 레이아웃과 읽기 순서 분석
- 수식·표·차트 등 복합 요소 파싱
- 문서 이미지에서 구조화된 결과를 생성
- 공식 모델 벤치마크 OmniDocBench v1.6에서 96.33%를 보고

96.33%는 문서 파싱 벤치마크 점수다. 칠판 손글씨 정확도나 강의 수식의 정확도와 동일한 수치가 아니다.

### Mac 로컬 실행 조건

PaddleOCR 공식 문서는 Apple Silicon용 설치·실행 경로를 별도로 제공한다. 공식 안내상 Apple M4에서 정확도와 속도를 검증했으며, M1~M3는 하드웨어별 호환성이 아직 확인되지 않았다고 안내한다. Apple Silicon 로컬 직접 실행은 PaddlePaddle 엔진 중심이며, 별도 VLM 서비스에는 MLX-VLM 백엔드를 사용할 수 있다. Docker 경로는 해당 안내에서 지원되지 않는다.

기존 Lecture2Doc의 `rapidocr + onnxruntime` 가상환경에 그대로 설치하는 것은 권하지 않는다. PaddleOCR-VL은 PaddlePaddle·PaddleOCR 의존성과 모델 파일이 필요하므로 별도 가상환경 또는 분리된 로컬 서비스로 운영하는 편이 안전하다.

## 3. PaddleOCR-VL 1.6을 기본 핵심 OCR로 바꿀 것인가?

현재는 **기본 OCR로 통째로 교체하지 않는 것**을 추천한다.

| 평가 항목 | 현재 RapidOCR + 손글씨 인식기 | PaddleOCR-VL 1.6 |
|---|---|---|
| 주요 목적 | 텍스트 영역 탐지와 문자 인식 | 문서 전체 레이아웃과 요소 파싱 |
| 한국어 손글씨 | 손글씨 데이터로 추가 학습한 인식기 적용 | 다국어 문서 파싱이 강점이나 칠판 손글씨 전용 모델은 아님 |
| 수식 구조 | 일반 문자 OCR 수준 | 수식 파싱 지원 |
| 모델 크기 | detector와 recognizer의 소형 ONNX 모델 | 모델 카드 약 1.93GB |
| 기존 파이프라인 호환 | RapidOCR API와 그대로 연결 | 별도 런타임·입출력 변환 필요 |
| CPU 처리 | 소형 ONNX 모델로 로컬 처리 | 가능하나 장당 지연·메모리 사용을 실측해야 함 |
| 권장 역할 | 기본 OCR | 어려운 판서·수식의 선택적 2차 분석 |

또한 공식 사용 문서는 VLM 구성 요소만 단독 호출하는 방식이 완성형 PaddleOCR-VL 파이프라인과 같지 않다고 강조한다. 레이아웃 분석과 VLM 인식을 함께 사용해야 하며, 불완전하게 연결하면 성능이 재현되지 않거나 환각성 문자가 늘 수 있다고 안내한다.

### 추천 도입 방식: 2단계 OCR

1. **기본 단계:** 현재처럼 PP-OCRv6 detector + 한국어 손글씨 recognizer로 모든 판서를 빠르게 읽는다.
2. **정밀 단계:** 수식이 많거나 OCR 신뢰도가 낮고, 사용자가 다시 확인하도록 표시한 페이지에만 PaddleOCR-VL 1.6 전체 파이프라인을 실행한다.
3. **결과 통합:** 기존 OCR과 PaddleOCR-VL 출력이 일치하는지 비교한다. 불일치는 자동으로 덮어쓰지 않고 검수 대상으로 남긴다.
4. **근거 보존:** 모델명·가중치 해시·원본 페이지 ID·입력 이미지 해시·출력 결과를 각각 보관한다.

이 방식은 일반 판서 처리 속도를 유지하면서 PaddleOCR-VL의 수식·레이아웃 강점을 필요한 곳에만 활용한다.

## 4. 검증 계획

한국어 손글씨 인식기의 실제 채택 여부는 아래 표본으로 판단한다.

- 실제 강의에서 한글 필기 30~50개 영역
- 분필·화이트보드, 작은 글씨, 흘림체, 영문 병기, 수학 기호를 포함
- 기존 PP-OCRv5와 새 손글씨 인식기를 동일한 detector·동일한 crop으로 비교
- CER, WER, 전체 단어 일치율, 낮은 신뢰도 영역 비율, 처리 시간 측정

PaddleOCR-VL 도입 여부는 수식·복잡한 판서 10~20페이지를 먼저 비교한다. 기본 OCR 대비 수식 인식과 구조 보존이 실제로 개선되고, Mac에서의 페이지당 처리 시간과 메모리 사용량이 감당할 만할 때만 2차 엔진으로 통합한다.

**현재 코드에는 한국어·영어 혼합 기본 인식기 복구, 선택형 손글씨 실험 모델, 기존 강의의 OCR 재실행 버튼을 반영했다. PaddleOCR-VL 1.6은 조사·평가 결과를 문서화했으며, 아직 애플리케이션 런타임에는 추가하지 않았다.** 실제 칠판 영상의 비교 테스트가 끝나기 전에는 기본 엔진 교체를 보류한다.

기존 강의의 OCR을 새 모델로 재실행하지 않은 상태에서는 과거 인식 결과가 그대로 보인다. 화면에서 모델명이 `korean-PP-OCRv5-mobile`인지 확인해야 기본 설정 적용 여부를 검증할 수 있다. 실험 모델을 선택했다면 `korean-handwriting-rec`으로 표시된다.


## 5. 모델 선택

기본값은 한국어·영어 혼합 텍스트를 지원하는 PP-OCRv5 Korean Mobile이다. 실험 손글씨 모델을 선택하려면 서버를 실행하기 전에 다음 환경 변수를 설정한다.

```bash
export LECTURE2DOC_KO_RECOGNIZER=handwriting
```

기본 모델로 돌아가려면 다음 설정을 해제한다.

```bash
unset LECTURE2DOC_KO_RECOGNIZER
```

모델 선택을 변경한 뒤에는 서버를 재시작하고 기존 강의의 **OCR 다시 실행**을 누른다. OCR 상태 줄에 실제 선택된 인식기 이름이 표시된다.
