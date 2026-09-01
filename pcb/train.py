"""PCB 假点过滤基线训练。

用法:
    python -m pcb.train                      # 按 config.yaml 的 input_mode 训一次
    python -m pcb.train --input-mode diff    # 指定输入表示
    python -m pcb.train --compare            # 四种输入表示各训一遍,出对比表
    python -m pcb.train --epochs 2 --quick   # 冒烟

数据只用「已标注」的成对样本(目前是测试集那 349 对),按板号分组切 train/val。
训练集那 1149 对还没分类,先用本脚本产出的权重跑 scripts/prelabel_pcb.py 做预标注,
人工只做纠正,不要从零手分。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import get_settings
from pcb.dataset import collect_pairs, describe, group_split, model_split, split_labeled
from pcb.labels import LABELS, NUM_CLASSES, OK_ID
from pcb.loader import INPUT_MODES, PairDataset
from pcb.metrics import (
    confusion_matrix,
    format_confusion,
    operating_point,
    per_class_recall,
)
from pcb.model import build_model, save_checkpoint


def _class_weights(labels: list[int], n: int) -> torch.Tensor:
    """逆频率权重。最少的类只有十几个样本,不加权基本学不动。"""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n * counts)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ok_probs: list[float] = []
    y_ng: list[int] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for x, ym, yb in loader:
        logit_b, logit_m = model(x.to(device))
        # 「假点」概率取二分类头的 class 0
        ok_probs.extend(torch.softmax(logit_b, dim=1)[:, 0].cpu().numpy().tolist())
        y_ng.extend(yb.numpy().tolist())
        y_true.extend(ym.numpy().tolist())
        y_pred.extend(logit_m.argmax(1).cpu().numpy().tolist())

    m = confusion_matrix(np.array(y_true), np.array(y_pred), NUM_CLASSES)
    op99 = operating_point(np.array(y_ng), np.array(ok_probs), 0.99)
    op100 = operating_point(np.array(y_ng), np.array(ok_probs), 1.0)
    return {
        "n": len(y_ng),
        "binary_acc": round(
            float(np.mean((np.array(ok_probs) < 0.5).astype(int) == np.array(y_ng))), 4
        ),
        "multi_acc": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        "op_recall99": op99,
        "op_recall100": op100,
        "confusion": m.tolist(),
        "per_class_recall": per_class_recall(m),
    }


def train_once(
    *,
    input_mode: str,
    epochs: int,
    holdout_model: str | None = None,
    verbose: bool = True,
) -> dict:
    s = get_settings()
    cfg = s.pcb
    device = s.device
    torch.manual_seed(cfg.seed)

    root = s.resolve(cfg.dataset_root)
    labeled, _unlabeled = split_labeled(collect_pairs(root))
    if not labeled:
        raise RuntimeError(
            f"{root} 下没有已标注样本。标注数据应放成 <类名>/xxx.jpg + xxx_T.jpg"
        )
    if holdout_model:
        train_pairs, val_pairs = model_split(labeled, holdout_model)
        split_desc = f"留出机种 {holdout_model}(跨机种泛化)"
    else:
        train_pairs, val_pairs = group_split(labeled, cfg.val_ratio, cfg.seed)
        split_desc = "按板号分组"
    if not val_pairs:
        raise RuntimeError("验证集为空,调大 pcb.val_ratio 或补充更多板号")

    ds_train = PairDataset(
        train_pairs, input_mode=input_mode, img_size=cfg.img_size, augment=True, seed=cfg.seed
    )
    ds_val = PairDataset(val_pairs, input_mode=input_mode, img_size=cfg.img_size, augment=False)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    from pcb.labels import label_id

    ids = [label_id(p.label) for p in train_pairs]
    w_multi = _class_weights(ids, NUM_CLASSES).to(device)
    w_bin = _class_weights([int(i != OK_ID) for i in ids], 2).to(device)

    model = build_model(input_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_b = nn.CrossEntropyLoss(weight=w_bin)
    loss_m = nn.CrossEntropyLoss(weight=w_multi)

    best: dict = {}
    best_key = (-1.0, -1.0)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, ym, yb in dl_train:
            x, ym, yb = x.to(device), ym.to(device), yb.to(device)
            opt.zero_grad()
            logit_b, logit_m = model(x)
            loss = loss_b(logit_b, yb) + cfg.multi_loss_weight * loss_m(logit_m, ym)
            loss.backward()
            opt.step()
            total += float(loss) * x.size(0)
        sched.step()
        res = evaluate(model, dl_val, device)
        op = res["op_recall99"]
        # 选模标准就是业务指标:NG 召回达标前提下过滤率最高
        key = (op["filter_rate"], op["ng_recall"])
        if key > best_key:
            best_key = key
            best = {**res, "epoch": ep}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                f"[{input_mode}] ep{ep:>3d} loss={total / max(1, len(ds_train)):.4f} "
                f"bin_acc={res['binary_acc']:.3f} multi_acc={res['multi_acc']:.3f} "
                f"NG召回99%时过滤率={op['filter_rate']:.3f}(thr={op['threshold']:.2f})"
            )

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - t0
    return {
        "input_mode": input_mode,
        "epochs": epochs,
        "split": split_desc,
        "holdout_model": holdout_model,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "train_boards": len({p.board for p in train_pairs}),
        "val_boards": len({p.board for p in val_pairs}),
        "elapsed_s": round(elapsed, 1),
        "best": best,
        "_model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PCB 假点过滤基线训练")
    ap.add_argument("--input-mode", choices=INPUT_MODES, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--compare", action="store_true", help="四种输入表示各训一遍出对比表")
    ap.add_argument(
        "--holdout-model",
        default=None,
        metavar="机种",
        help="留出整个机种当验证集(如 Pros),用来测换型泛化;默认按板号分组切",
    )
    ap.add_argument("--quick", action="store_true", help="冒烟:不写权重")
    args = ap.parse_args()

    s = get_settings()
    epochs = args.epochs or s.pcb.epochs
    modes = list(INPUT_MODES) if args.compare else [args.input_mode or s.pcb.input_mode]

    root = s.resolve(s.pcb.dataset_root)
    snapshot = describe(collect_pairs(root))
    print("数据概况:", json.dumps(snapshot, ensure_ascii=False, indent=2))

    results = []
    for mode in modes:
        r = train_once(input_mode=mode, epochs=epochs, holdout_model=args.holdout_model)
        results.append(r)

    print(f"\n=== 输入表示对比(切分:{results[0]['split']};NG 召回 ≥99% 时的假点过滤率)===")
    print(f"{'input_mode':<12}{'过滤率':>10}{'阈值':>8}{'漏检':>6}{'多分类acc':>10}{'耗时s':>8}")
    for r in results:
        op = r["best"]["op_recall99"]
        print(
            f"{r['input_mode']:<12}{op['filter_rate']:>10.3f}{op['threshold']:>8.2f}"
            f"{op['missed_ng']:>6d}{r['best']['multi_acc']:>10.3f}{r['elapsed_s']:>8.1f}"
        )

    winner = max(results, key=lambda r: r["best"]["op_recall99"]["filter_rate"])
    print(f"\n最佳输入表示: {winner['input_mode']}")
    print("\n混淆矩阵(行=真值,列=预测):")
    print(format_confusion(np.array(winner["best"]["confusion"])))

    report = {
        "labels": list(LABELS),
        "data": snapshot,
        "runs": [{k: v for k, v in r.items() if k != "_model"} for r in results],
        "winner": winner["input_mode"],
    }
    out = s.artifacts_path / "pcb"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告: {out / 'train_report.json'}")

    if args.quick:
        print("--quick:跳过写权重")
        return
    if args.holdout_model:
        # 留出机种是评估用的,训练集少了一整个机种,不该拿它当交付权重
        print(f"--holdout-model {args.holdout_model}:这是泛化评估,不写权重")
        return
    ckpt = save_checkpoint(
        winner["_model"],
        s.resolve(s.pcb.model_path),
        input_mode=winner["input_mode"],
        img_size=s.pcb.img_size,
        metrics=winner["best"],
    )
    print(f"权重: {ckpt}")


if __name__ == "__main__":
    main()
