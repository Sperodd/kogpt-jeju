# KoGPT-Jeju

**KoGPT2를 제주어 문장으로 파인튜닝하고, 원본 모델과 생성 결과를 웹에서 비교하는 프로젝트입니다.**

한국어 사전학습 모델에 제주어 발화 데이터를 추가 학습하여 문장 이어쓰기 방식이 어떻게 달라지는지 살펴봅니다. PyTorch와 Transformers로 학습하고, FastAPI와 HTML로 구성한 간단한 화면에서 두 모델을 비교합니다.

## 주요 기능

- 제주어 문장 데이터로 KoGPT2 전체 파라미터 파인튜닝
- 원본 모델과 학습 후 모델의 검증 손실 기록
- 검증 손실이 가장 낮은 에포크의 모델 저장
- 동일한 입력·난수 시드·생성 설정으로 두 모델의 이어쓰기 결과 비교
- RunPod에서 설치, 학습, 서버 실행을 위한 Bash 스크립트 제공

## 실행 예시

입력한 문장 뒤에 이어지는 **생성 부분만** 화면에 표시합니다.

![원본 KoGPT2와 제주어 파인튜닝 모델의 이어쓰기 비교](results/results3.png)

<details>
<summary>다른 입력의 비교 결과</summary>

![제주어 문장 이어쓰기 비교 예시](results/results1.png)

</details>

예시 화면에서는 추가 학습 모델이 원본과 다른 어미와 짧은 발화를 생성하는 모습을 볼 수 있습니다.

## 기술 구성

| 구분 | 사용 기술 |
| --- | --- |
| 기반 모델 | `skt/kogpt2-base-v2` |
| 학습 | PyTorch, Hugging Face Transformers |
| 서버 | FastAPI, Uvicorn |
| 웹 화면 | HTML, CSS, JavaScript |
| 실행 환경 | Linux / RunPod GPU, Bash |

## 프로젝트 구조

```text
kogpt-jeju/
├── model.py          # 데이터 분할, 학습, 검증 및 모델 저장
├── common.py         # 모델 경로와 토크나이저 설정
├── app.py            # 모델 로딩 및 비교 API
├── index.html        # 두 모델의 생성 결과 비교 화면
├── data_loader.py    # 원천 JSON을 문장 TXT로 변환하는 기존 코드
├── setup.sh          # 환경 설치·학습·서버 실행
├── requirements.txt  # Python 의존성
├── results/          # 웹 비교 결과 이미지
├── sentences.txt     # 학습 문장 (별도 준비, Git 제외)
└── outputs/          # 학습 모델과 보고서 (실행 시 생성, Git 제외)
```

## 데이터 준비

프로젝트 루트에 `sentences.txt`를 배치합니다. UTF-8 인코딩으로 **한 줄에 제주어 문장 하나**를 저장합니다. 

예시:
```text
예전에 겅
아 겅하니까이 예초 할땐 조심해야 돼
겅허고 예초 벌초하다그네
어 그거 튀겨그네
돌멩이 해 가지고 실명된사람 이서
아그러니까 뉴스에도 많이 나오잖아 뭐막 짤리고 막 영 이추룩 한다고 거난
거난 제주도는 워낙 또 돌멩이가 많아그네
겅하니까
겅하니깐
데리고 가고 싶어그네 할아버지가이
```
데이터는 aihub의 **한국어 방언 발화(제주도)**을 사용했습니다. 

링크: [한국어 방언 발화(제주도)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=121)

`setup.sh`와 `model.py`는 TXT를 자동으로 생성하지 않습니다. 기존 TXT가 있으면 그대로 사용하며, 원천 JSON에서 다시 생성할 경우 `data_loader.py`의 데이터 경로와 `make_txt()` 호출을 별도로 준비해야 합니다.

이 데이터는 표준어–제주어 번역 쌍이나 질문–답변 쌍이 아닙니다. 모델은 **다음 토큰을 예측하는 문장 이어쓰기**를 학습합니다.

## 빠른 시작

아래 명령은 Linux / RunPod 터미널 기준입니다. Python 3.10 이상과 CUDA를 사용할 수 있는 PyTorch GPU 환경을 준비합니다. 프로젝트 폴더를 `/workspace/kogpt-jeju`에 배치하고, `sentences.txt`를 별도로 업로드합니다.

### 1. 환경 설치

```bash
cd /workspace/kogpt-jeju
bash setup.sh
```

설치 및 최초 기반 모델 다운로드에는 인터넷 연결이 필요합니다. 실행 로그에서 `CUDA: True`를 확인합니다. CUDA가 없으면 CPU로 동작합니다.

### 2. 학습

```bash
bash setup.sh train
```

기본 모델은 `outputs/jeju-kogpt2/`에 저장됩니다. 기존 결과가 있는 폴더는 덮어쓰지 않습니다.

배치 크기와 저장 위치를 바꾸려면 직접 학습 스크립트를 실행합니다.

```bash
source .venv/bin/activate
python model.py --batch-size 32 --epochs 3 --output outputs/jeju-b32
```

배치 32는 변경 예시이며 GPU 메모리에 맞게 조절합니다. `setup.sh train`의 기본 배치는 16입니다.

### 3. 웹 서버 실행

기본 경로의 모델을 사용할 때:

```bash
bash setup.sh serve
```

다른 경로에 저장한 모델을 사용할 때:

```bash
MODEL_DIR=outputs/jeju-b32 bash setup.sh serve
```

RunPod에서 HTTP 포트 **8000**을 공개하고 해당 HTTP 서비스에 접속합니다. 서버 터미널에 표시되는 **접속 키**를 웹 화면에 입력한 뒤 문장을 넣으면 두 모델의 결과를 확인할 수 있습니다.

서버는 `0.0.0.0:8000`에서 실행되며 두 모델을 시작 시 한 번씩 로딩합니다. 접속 키는 자동 생성되고, 12자 이상의 `APP_TOKEN` 환경변수로 지정할 수도 있습니다. 현재 인증과 단일 요청 처리는 실습용 구성입니다.

처음부터 설치 → 학습 → 서버 실행을 이어서 수행하려면 `bash setup.sh all`을 사용합니다. 이미 학습한 모델이 있다면 `serve`만 실행합니다.

### 작은 데이터로 동작 확인

```bash
source .venv/bin/activate
python model.py --max-samples 200 --epochs 1 --output outputs/smoke
MODEL_DIR=outputs/smoke bash setup.sh serve
```

이 설정은 실행 흐름 확인용이며 전체 데이터 학습 결과와 구분해야 합니다.

## 학습 방식

1. 공백을 정리하고 빈 문장과 완전히 동일한 문장을 제거합니다.
2. 고정 시드로 섞은 후 문장 단위로 학습 약 90%, 검증 약 10%를 나눕니다.
3. 문장을 토큰화하고 문장 종료 토큰(EOS)을 붙입니다.
4. 패딩 토큰(PAD)은 손실 계산에서 제외하고 EOS는 학습에 포함합니다.
5. 원본 모델의 검증 손실을 측정한 뒤 전체 파라미터를 추가 학습합니다.
6. 에포크별 손실을 기록하고 학습한 에포크 중 검증 손실이 가장 낮은 모델을 저장합니다.

| 설정 | 기본값 |
| --- | --- |
| 에포크 | 3 |
| 배치 크기 | 8 |
| 최대 시퀀스 길이 | 128토큰, EOS 포함 |
| 학습률 | 0.00005 |
| 옵티마이저 | AdamW |
| 수치 정밀도 | FP32 |
| 난수 시드 | 42 |

`--data`, `--output`, `--epochs`, `--batch-size`, `--max-length`, `--learning-rate`, `--max-samples`, `--seed` 옵션으로 설정을 변경할 수 있습니다.

## 결과 확인과 한계

학습 폴더의 `training_report.json`에는 원본 검증 손실, 에포크별 학습·검증 손실, 최적 에포크, 데이터 수와 실행 설정이 기록됩니다.

- 검증 손실은 다음 토큰 예측 오차이며 제주어 정확도나 번역 점수가 아닙니다.
- TXT에는 화자·대화 ID가 없어 같은 대화의 비슷한 문장이 학습과 검증에 나뉠 수 있습니다. 독립적인 화자·대화에 대한 평가로 해석하지 않습니다.
- 검증 데이터는 에포크 선택에도 사용하므로 별도 테스트 데이터의 최종 성능과 구분합니다.
- 웹에서는 여러 입력과 시드로 제주어 표현, 문맥 연결, 반복, 조기 종료를 비교할 수 있습니다. 생성 문장의 정확성과 자연스러움은 별도 검토가 필요합니다.

## 모델 보관

모델 폴더에는 가중치뿐 아니라 설정과 토크나이저도 들어 있습니다. 폴더 전체를 보관해야 다시 불러올 수 있습니다.

```bash
# 배치 32 예시 경로로 학습한 경우
tar -czf jeju-b32.tar.gz -C outputs jeju-b32
```

압축 파일을 로컬로 내려받아 정상적으로 풀리는지 확인한 후 Pod를 정리합니다. 이후 폴더를 복원하고 `MODEL_DIR`로 지정하면 재학습 없이 서버에서 사용할 수 있습니다.

## 문제 해결

| 상황 | 확인할 내용 |
| --- | --- |
| CUDA 메모리 부족 | `--batch-size`를 줄여 새 출력 경로로 실행 |
| 모델을 찾을 수 없음 | 학습 완료 여부와 `MODEL_DIR` 확인 |
| 기존 출력 폴더 오류 | `--output`에 새로운 경로 지정 |
| 웹 요청 시간 초과 | 생성 토큰 수를 줄이고 GPU 사용 여부 확인 |
| 모델 변경이 반영되지 않음 | 새 경로를 지정한 후 서버 재시작 |
| Windows 가상환경 오류 | Linux 업로드에서 `.venv` 제외 후 환경 설치 |

## 데이터 및 모델 이용

학습 데이터, 가중치, 가상환경은 `.gitignore`에서 제외합니다. Git으로 프로젝트를 가져온 경우 데이터 또는 학습된 모델을 별도로 준비해야 합니다. 원천 데이터와 기반 모델의 이용 조건은 각각 해당 배포처에서 확인합니다.

- [KoGPT2 기반 모델 및 모델 카드](https://huggingface.co/skt/kogpt2-base-v2)
- [RunPod HTTP 포트 설정](https://docs.runpod.io/pods/configuration/expose-ports)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
