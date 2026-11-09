#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE="${1:-install}"
case "$MODE" in
  install|experiment|serve) ;;
  *) echo "사용법: bash setup.sh [install|experiment|serve]"; exit 1 ;;
esac
if [[ -d .venv && ! -f .venv/bin/activate ]]; then
  echo "Windows .venv는 Linux에서 사용할 수 없습니다. 업로드에서 .venv를 제외하세요."
  exit 1
fi
if [[ ! -f .venv/bin/activate ]]; then
  python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
if [[ "$MODE" == install ]]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3 (KoGPT-Jeju)"
fi
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU 실행: 학습은 오래 걸릴 수 있습니다.")'
if [[ "$MODE" == experiment ]]; then
  python experiment.py
fi
if [[ "$MODE" == serve ]]; then
  if [[ -z "${APP_TOKEN:-}" ]]; then
    APP_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"
    export APP_TOKEN
    printf '\n웹 화면에 입력할 접속 키: %s\n\n' "$APP_TOKEN"
  else
    echo "환경변수 APP_TOKEN을 접속 키로 사용합니다."
  fi
  exec python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
fi
echo "다음 명령: bash setup.sh experiment / bash setup.sh serve"
