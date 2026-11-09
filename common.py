import os
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
BASE_MODEL = "skt/kogpt2-base-v2"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(ROOT / "outputs/jeju-kogpt2")))
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT / MODEL_DIR


def load_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(
        source, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>",
    )
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("PAD와 EOS는 서로 달라야 합니다.")
    return tokenizer
