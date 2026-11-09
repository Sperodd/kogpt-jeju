"""완료된 실험에서 표/그래프/결론과 출력이 포함된 제출용 노트북 생성."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NOTEBOOK = "과제4-202435119-한여원.ipynb"


def summarize(report):
    rows = []
    for method in ("base", "full", "lora"):
        item = report["models"][method]
        rows.append({
            "method": method, "validation_loss": item["validation"]["loss"],
            "validation_ppl": item["validation"]["perplexity"],
            "test_loss": item["test"]["loss"], "test_ppl": item["test"]["perplexity"],
            "train_seconds": item["train_seconds"], "fit_seconds": item["fit_seconds"],
            "trainable_parameters": item["trainable_parameters"],
            "total_parameters": item["total_parameters"],
            "best_epoch": item.get("best_epoch", 0),
        })
    return rows


def conclusion(report):
    m = report["models"]
    base = m["base"]["test"]["perplexity"]
    full, lora = m["full"]["test"]["perplexity"], m["lora"]["test"]["perplexity"]
    def effect(name, value):
        direction = "감소" if value < base else "증가" if value > base else "변화 없음"
        return f"{name}의 테스트 PPL은 {value:.3f}이며 원본 {base:.3f} 대비 {abs(value/base-1)*100:.2f}% {direction}했다."
    ratio = m["lora"]["trainable_parameters"] / m["full"]["trainable_parameters"] * 100
    return [
        effect("전체 미세조정", full) + " 이 비교는 같은 토크나이저와 토큰화 조건의 도메인 예측 오차를 나타낸다.",
        effect("LoRA", lora) + " 미세조정 방식에 따라 도메인 적응의 정도가 달랐다." if full != lora else effect("LoRA", lora),
        f"LoRA의 학습 파라미터 수는 전체 미세조정의 {ratio:.3f}%이다. 순수 학습 시간은 Full {m['full']['train_seconds']:.1f}초, LoRA {m['lora']['train_seconds']:.1f}초로 측정되었으며, 파라미터 감소율과 시간 감소율은 동일하지 않다.",
        "동일 프롬프트 5개의 생성 결과를 아래 출력으로 비교한다. PPL만으로 자연스러운 제주어 사용이나 사실 정확성을 판정할 수 없으며, 짧은 응답·반복·문맥 연결은 별도 정성 검토가 필요하다.",
        "대화 파일 간 분리와 완전 동일 문장 제거를 적용했지만, 파일 간 화자 중복이나 유사 문장까지 배제한 것은 아니다. 단일 시드·선택한 표본·학습 설정에서의 결과이며 독립 화자 전체에 대한 성능으로 일반화하지 않는다.",
    ]


def build_report(run_dir, execute_notebook=True):
    run_dir = Path(run_dir).resolve()
    report = json.loads((run_dir / "experiment_report.json").read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError("Full/LoRA 학습과 최종 테스트를 모두 완료한 보고서가 필요합니다.")
    rows = summarize(report)
    with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, method in zip(axes, ("full", "lora")):
        history = report["models"][method]["history"]
        x = [r["epoch"] for r in history]
        ax.plot(x, [r["train_loss"] for r in history], "o-", label="Train")
        ax.plot(x, [r["validation_loss"] for r in history], "o-", label="Validation")
        ax.axhline(report["models"]["base"]["validation"]["loss"], ls="--", color="gray", label="Base validation")
        ax.set(title=method.upper(), xlabel="Epoch", ylabel="Token-weighted cross entropy", xticks=x)
        ax.legend()
        ax.grid(alpha=.25)
    fig.savefig(run_dir / "loss_curves.png", dpi=160)
    plt.close(fig)
    lines = conclusion(report)
    (run_dir / "conclusion.md").write_text("# 측정 결과에 근거한 결론\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    if execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        nb = nbformat.read(ROOT / NOTEBOOK, as_version=4)
        old = os.environ.get("JEJU_RUN_DIR")
        os.environ["JEJU_RUN_DIR"] = str(run_dir)
        try:
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        finally:
            if old is None:
                os.environ.pop("JEJU_RUN_DIR", None)
            else:
                os.environ["JEJU_RUN_DIR"] = old
        nbformat.write(nb, run_dir / NOTEBOOK)
        print(f"제출용 실행 노트북: {run_dir / NOTEBOOK}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/assignment4")
    p.add_argument("--skip-notebook", action="store_true")
    a = p.parse_args()
    build_report(a.run_dir, execute_notebook=not a.skip_notebook)
