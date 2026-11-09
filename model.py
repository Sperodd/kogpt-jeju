"""제주어 문장으로 KoGPT2 추가 학습: python model.py"""
import argparse
import hashlib
import json
import math
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, set_seed
from common import ROOT, BASE_MODEL, MODEL_DIR, load_tokenizer


def read_sentences(path):
    lines = [" ".join(line.split()) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return list(dict.fromkeys(line for line in lines if line))


def encode_sentences(lines, tokenizer, max_length):
    rows = []
    for line in lines:
        ids = tokenizer(line, add_special_tokens=False, truncation=True,
                        max_length=max_length - 1)["input_ids"]
        if ids:
            ids = ids + [tokenizer.eos_token_id]
            rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return rows


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += model(**batch).loss.item() * n
        count += n
    if not count:
        raise ValueError("검증 토큰이 없습니다.")
    return total / count


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2 or args.learning_rate <= 0:
        raise ValueError("학습 설정값을 확인하세요.")
    if args.max_samples != 0 and args.max_samples < 20:
        raise ValueError("max-samples는 0(전체) 또는 20 이상이어야 합니다.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"기존 결과 보호: 다른 --output 경로를 사용하세요: {args.output}")
    set_seed(args.seed)
    lines = read_sentences(args.data)
    random.Random(args.seed).shuffle(lines)
    if args.max_samples:
        lines = lines[:args.max_samples]
    if len(lines) < 20:
        raise ValueError("중복 제거 후 문장이 최소 20개 필요합니다.")
    n_val = max(1, len(lines) // 10)
    tokenizer = load_tokenizer(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max-length가 모델의 문맥 길이를 초과합니다.")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache = False
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_rows = encode_sentences(lines[n_val:], tokenizer, args.max_length)
    val_rows = encode_sentences(lines[:n_val], tokenizer, args.max_length)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "base_model": BASE_MODEL, "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "train_sentences": len(train_rows), "validation_sentences": len(val_rows),
        "seed": args.seed, "max_length": args.max_length, "batch_size": args.batch_size,
        "epochs": args.epochs, "learning_rate": args.learning_rate, "device": str(device),
        "evaluation_note": "문장 단위 검증. 화자/대화 독립 평가나 제주어 정확도가 아님.",
        "base_validation_loss": evaluate(model, val_loader, device), "history": [],
    }
    print(f"device={device}, train={len(train_rows)}, validation={len(val_rows)}", flush=True)
    print(f"base validation loss={report['base_validation_loss']:.4f}", flush=True)
    best = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("학습 손실이 유한하지 않습니다.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = int((batch["labels"][:, 1:] != -100).sum())
            total += loss.item() * n
            count += n
            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        val_loss = evaluate(model, val_loader, device)
        if not math.isfinite(val_loss):
            raise RuntimeError("검증 손실이 유한하지 않습니다.")
        report["history"].append({"epoch": epoch, "train_loss": total / count, "validation_loss": val_loss})
        print(f"epoch={epoch} validation loss={val_loss:.4f}", flush=True)
        if val_loss < best:
            best = val_loss
            report["best_epoch"] = epoch
            report["best_validation_loss"] = best
            model.config.use_cache = True
            model.save_pretrained(args.output, safe_serialization=True)
            tokenizer.save_pretrained(args.output)
            model.config.use_cache = False
        (args.output / "training_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {args.output}", flush=True)


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "sentences.txt")
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
