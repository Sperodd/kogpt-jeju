"""원천 JSON을 대화 단위 train/validation/test로 분할합니다."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_utterance(row):
    if row.get("form") == row.get("standard_form"):
        return None, "same_as_standard"
    words = row.get("eojeolList") or []
    if any("{" in str(w.get("eojeol", "")) for w in words):
        return None, "anonymization_marker"
    cleaned = []
    for item in words:
        word = str(item.get("eojeol") or "").split("/")[0]
        word = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣]", "", word)
        if word:
            cleaned.append(word)
    if len(cleaned) < 2:
        return None, "fewer_than_two_words"
    return " ".join(cleaned), None


def prepare(source, output, max_files=300, seed=42):
    source, output = Path(source), Path(output)
    if max_files < 0:
        raise ValueError("max-files: 0=전체, 또는 양수")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"기존 데이터 보호: 새 --output 경로를 쓰세요: {output}")
    files = sorted(source.glob("*.json"))
    if len(files) < 10:
        raise ValueError(f"JSON 파일이 최소 10개 필요합니다: {source}")
    rng = random.Random(seed)
    rng.shuffle(files)
    selected = files[:max_files] if max_files else files
    if len(selected) < 10:
        raise ValueError("선택된 대화가 최소 10개 필요합니다.")
    n_test = n_val = max(1, len(selected) // 10)
    assignment = {}
    for i, path in enumerate(selected):
        assignment[path.name] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    rows = {name: [] for name in ("train", "validation", "test")}
    stats, manifest, ids = Counter(), [], set()
    # 학습을 먼저 정제한 뒤 검증/테스트에서 기존과 동일한 문장을 제외합니다.
    seen = set()
    for split in rows:
        for path in selected:
            if assignment[path.name] != split:
                continue
            raw = path.read_bytes()
            doc = json.loads(raw.decode("utf-8-sig"))
            group = str(doc.get("id") or path.stem)
            if group in ids:
                raise ValueError(f"중복 대화 ID: {group}")
            ids.add(group)
            kept, reasons = 0, Counter()
            for item in doc.get("utterance", []):
                stats["raw_utterances"] += 1
                text, reason = clean_utterance(item)
                if reason:
                    reasons[reason] += 1
                    continue
                if text in seen:
                    reasons["exact_duplicate"] += 1
                    continue
                seen.add(text)
                rows[split].append({"text": text, "group_id": group,
                                    "utterance_id": str(item.get("id", ""))})
                kept += 1
            stats.update(reasons)
            manifest.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                             "group_id": group, "split": split, "kept": kept,
                             "removed": dict(reasons)})
            if len(manifest) % 50 == 0:
                print(f"JSON {len(manifest)}/{len(selected)}", flush=True)
    if any(not data for data in rows.values()):
        raise ValueError("빈 split이 있습니다. 다른 시드 또는 더 많은 파일로 준비하세요.")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, data in rows.items():
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data), encoding="utf-8")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "source": "AI Hub 한국어 방언 발화(제주도), 비식별화완료 제주도 학습용데이터 1",
        "source_url": "https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=121",
        "available_json_files": len(files), "selected_json_files": len(selected),
        "selection": "정렬한 JSON 목록을 seed로 섞은 후 max_files개 선택 (0=전체)",
        "seed": seed, "raw_and_removed_counts": dict(stats),
        "split_sentences": {s: len(v) for s, v in rows.items()},
        "split_documents": dict(Counter(item["split"] for item in manifest)),
        "split_nonempty_documents": {s: len({r["group_id"] for r in v}) for s, v in rows.items()},
        "split_sha256": hashes,
        "split_note": "대화 JSON 단위 80/10/10. 문장 비율은 달라질 수 있음. 파일 간 동일 화자는 확인되지 않아 화자 독립을 보장하지 않음.",
        "deduplication": "train→validation→test 순으로 정제 후 완전 동일 문장 제거",
        "manifest": manifest,
    }
    write_json(output / "data_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "prepared")
    p.add_argument("--max-files", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare(args.source, args.output, args.max_files, args.seed)
