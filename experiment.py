"""동일 split으로 Base / Full fine-tuning / LoRA 비교. RunPod GPU에서 실행."""
import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, set_seed
from common import ROOT, BASE_MODEL, load_tokenizer
from prepare_data import write_json

PROMPTS = [
    "오늘 바당에 가 보니까",
    "우리 어멍이 아침에",
    "친구하고 시장에 가서",
    "어릴 때 우리 동네에서는",
    "비가 많이 오는 날에는",
]
GENERATION = dict(max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9,
                  top_k=50, repetition_penalty=1.1, no_repeat_ngram_size=3)


def load_data(directory):
    report = json.loads((directory / "data_report.json").read_text(encoding="utf-8"))
    rows = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["split_sha256"][split]:
            raise ValueError(f"{split} 해시 불일치: 준비한 파일이 바뀌었습니다.")
        rows[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows[split]) != report["split_sentences"][split] or not rows[split]:
            raise ValueError(f"{split} 건수 불일치 또는 빈 데이터")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for key in ("group_id", "text"):
            if {r[key] for r in rows[a]} & {r[key] for r in rows[b]}:
                raise ValueError(f"{a}/{b}의 {key} 중복")
    return rows, report


def encode(rows, tokenizer, max_length):
    encoded, truncated = [], 0
    for row in rows:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError("토큰이 없는 문장")
        truncated += len(ids) > max_length - 2
        # BOS를 넣어 첫 실제 토큰도 예측 대상에 포함. 모든 모델에 동일하게 적용.
        ids = [tokenizer.bos_token_id] + ids[:max_length - 2] + [tokenizer.eos_token_id]
        encoded.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return encoded, truncated


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total_nll, tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum().item())
        loss = float(model(**batch).loss)
        if not math.isfinite(loss):
            raise RuntimeError("유한하지 않은 평가 손실")
        total_nll += loss * n
        tokens += n
    if not tokens:
        raise ValueError("평가 토큰이 없습니다.")
    loss = total_nll / tokens
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": tokens}


@torch.inference_mode()
def generate_examples(model, tokenizer, device, seed):
    model.eval()
    results = []
    for prompt in PROMPTS:
        ids = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False)["input_ids"]
        inputs = torch.tensor([ids], device=device)
        set_seed(seed)
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, **GENERATION)
        results.append({"prompt": prompt, "continuation": tokenizer.decode(
            output[0, len(ids):], skip_special_tokens=True)})
    return results


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 3:
        raise ValueError("epochs/batch-size/max-length 확인")
    if min(args.full_lr, args.lora_lr) <= 0:
        raise ValueError("학습률은 양수여야 합니다.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"기존 실험 보호: 새 --output 경로를 사용하세요: {args.output}")
    raw, data_report = load_data(args.data)
    tokenizer = load_tokenizer(args.base_model)
    data, truncation = {}, {}
    for split, rows in raw.items():
        data[split], truncation[split] = encode(rows, tokenizer, args.max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    def loader(split, shuffle=False):
        # 별도 RNG로 두 방식의 배치 순서까지 같게 고정합니다.
        return DataLoader(data[split], batch_size=args.batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(args.seed),
                          collate_fn=collator, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def fresh_model():
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
        if args.max_length > model.config.max_position_embeddings:
            raise ValueError("max-length가 모델의 최대 문맥 길이보다 큽니다.")
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        return model.to(device)
    set_seed(args.seed)
    base = fresh_model()
    total_parameters = sum(p.numel() for p in base.parameters())
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "base_model": args.base_model,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": str(device),
                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "data": data_report, "truncated_sentences": truncation,
        "evaluation": "문장별 BOS + 최대 126 내용토큰 + EOS. shifted non-PAD 토큰 가중 평균 NLL의 exp. 같은 test를 최종 1회 평가.",
        "generation_settings": GENERATION, "prompts": PROMPTS, "models": {},
        "timing_note": "train_seconds=학습 배치의 forward/backward/optimizer 및 데이터 로딩. fit_seconds=학습+에포크 검증+체크포인트 저장. 다운로드/토큰화/최종평가 제외.",
    }
    report["models"]["base"] = {
        "validation": evaluate(base, loader("validation"), device),
        "trainable_parameters": 0, "total_parameters": total_parameters,
        "train_seconds": 0.0, "fit_seconds": 0.0, "history": [],
    }
    del base
    release()
    write_json(args.output / "experiment_report.json", report)
    for method in ("full", "lora"):
        print(f"===== {method.upper()} =====", flush=True)
        set_seed(args.seed)
        model = fresh_model()  # FULL의 결과로 LoRA를 시작하지 않습니다.
        lr = args.full_lr if method == "full" else args.lora_lr
        if method == "lora":
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["c_attn"], fan_in_fan_out=True, bias="none"))
        model.config.use_cache = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=lr, weight_decay=0.01)
        train_loader, val_loader = loader("train", True), loader("validation")
        item = {"trainable_parameters": trainable,
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "learning_rate": lr, "history": [], "train_seconds": 0.0}
        if method == "lora":
            item["lora_config"] = {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["c_attn"]}
        best, destination = math.inf, args.output / method
        sync(device)
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            # LoRA 초기화의 RNG 소비가 데이터 순서에 영향을 주지 않도록 loader RNG 별도 사용.
            sync(device)
            epoch_started = time.perf_counter()
            total, count = 0.0, 0
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("학습 loss가 유한하지 않습니다.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = int((batch["labels"][:, 1:] != -100).sum().item())
                total += float(loss.detach()) * n
                count += n
                if step % 50 == 0 or step == len(train_loader):
                    print(f"{method} epoch={epoch} step={step}/{len(train_loader)} loss={float(loss.detach()):.4f}", flush=True)
            sync(device)
            elapsed = time.perf_counter() - epoch_started
            item["train_seconds"] += elapsed
            val = evaluate(model, val_loader, device)
            item["history"].append({"epoch": epoch, "train_loss": total / count,
                                    "validation_loss": val["loss"], "validation_ppl": val["perplexity"],
                                    "train_seconds": elapsed})
            print(f"{method} validation PPL={val['perplexity']:.3f}", flush=True)
            if val["loss"] < best:
                best = val["loss"]
                item["best_epoch"], item["validation"] = epoch, val
                model.config.use_cache = True
                model.save_pretrained(destination, safe_serialization=True)
                tokenizer.save_pretrained(destination)
                # 웹 서버가 BOS 설정을 읽도록 저장합니다.
                write_json(destination / "experiment_meta.json", {"method": method, "prepend_bos": True})
                model.config.use_cache = False
            item["fit_seconds"] = time.perf_counter() - started
            report["models"][method] = item
            write_json(args.output / "experiment_report.json", report)
        del optimizer, model, train_loader, val_loader
        release()
    # 두 방식의 에포크 선택 완료 후에만 테스트합니다.
    for method in ("base", "full", "lora"):
        print(f"Final test: {method}", flush=True)
        if method == "full":
            model = AutoModelForCausalLM.from_pretrained(args.output / "full").to(device)
        else:
            model = fresh_model()
            if method == "lora":
                model = PeftModel.from_pretrained(model, args.output / "lora").to(device)
        model.config.use_cache = True
        report["models"][method]["test"] = evaluate(model, loader("test"), device)
        report["models"][method]["generations"] = generate_examples(model, tokenizer, device, args.seed)
        write_json(args.output / "experiment_report.json", report)
        del model
        release()
    report["status"] = "complete"
    write_json(args.output / "experiment_report.json", report)
    from make_report import build_report
    build_report(args.output)
    print(f"완료: {args.output}", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "prepared")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/assignment4")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--full-lr", type=float, default=5e-5)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    run(p.parse_args())
