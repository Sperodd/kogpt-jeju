"""서버: python -m uvicorn app:app --host 0.0.0.0 --port 8000"""
import logging
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, set_seed
from common import ROOT, BASE_MODEL, MODEL_DIR, load_tokenizer


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    max_new_tokens: int = Field(default=60, ge=8, le=100)
    seed: int = Field(default=42, ge=0, le=2147483647)


class Models:
    def __init__(self):
        if not (MODEL_DIR / "config.json").is_file() and not (MODEL_DIR / "adapter_config.json").is_file():
            raise FileNotFoundError(f"먼저 python model.py로 학습하세요. 모델 경로: {MODEL_DIR}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        metadata = MODEL_DIR / "experiment_meta.json"
        self.prepend_bos = json.loads(metadata.read_text(encoding="utf-8")).get("prepend_bos", False) if metadata.is_file() else False
        self.items = {}
        for name, source in [("base", BASE_MODEL), ("finetuned", MODEL_DIR)]:
            tokenizer = load_tokenizer(source)
            if name == "finetuned" and (MODEL_DIR / "adapter_config.json").is_file():
                from peft import PeftConfig, PeftModel
                adapter = PeftConfig.from_pretrained(MODEL_DIR)
                model = AutoModelForCausalLM.from_pretrained(adapter.base_model_name_or_path)
                model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(model, MODEL_DIR)
            else:
                model = AutoModelForCausalLM.from_pretrained(source)
            model.resize_token_embeddings(len(tokenizer))
            self.items[name] = (tokenizer, model.to(self.device).eval())

    @torch.inference_mode()
    def compare(self, request):
        result = {"prompt": request.prompt, "device": self.device}
        for name, (tokenizer, model) in self.items.items():
            inputs = tokenizer(request.prompt, add_special_tokens=False, return_tensors="pt", return_token_type_ids=False)
            if self.prepend_bos:
                inputs["input_ids"] = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), inputs["input_ids"]], dim=1)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            length = inputs["input_ids"].shape[1]
            if not 1 <= length <= 128:
                raise ValueError("입력은 토큰 기준 1~128개여야 합니다. 문장을 짧게 줄여주세요.")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            set_seed(request.seed)
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model.generate(
                **inputs, max_new_tokens=request.max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.9, top_k=50, repetition_penalty=1.1,
                no_repeat_ngram_size=3, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            result[name] = {
                "continuation": tokenizer.decode(output[0, length:], skip_special_tokens=True),
                "seconds": round(time.perf_counter() - start, 2),
            }
        return result


@asynccontextmanager
async def lifespan(app):
    token = os.environ.get("APP_TOKEN", "")
    if len(token) < 12:
        raise RuntimeError("APP_TOKEN을 12자 이상으로 설정하거나 bash setup.sh serve로 실행하세요.")
    app.state.token = token
    app.state.models = Models()
    app.state.lock = threading.Lock()
    yield
    del app.state.models


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare")
def compare(request: CompareRequest, x_app_token: str = Header(default="")):
    if not secrets.compare_digest(x_app_token.encode(), app.state.token.encode()):
        raise HTTPException(401, "접속 키가 올바르지 않습니다.")
    request.prompt = request.prompt.strip()
    if not request.prompt:
        raise HTTPException(422, "문장을 입력하세요.")
    if not app.state.lock.acquire(blocking=False):
        raise HTTPException(429, "다른 요청을 처리 중입니다. 잠시 후 다시 시도하세요.")
    try:
        return app.state.models.compare(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception("문장 생성 실패")
        raise HTTPException(500, "생성에 실패했습니다. 서버 터미널을 확인하세요.") from exc
    finally:
        app.state.lock.release()
