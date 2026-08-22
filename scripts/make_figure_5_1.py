#!/usr/bin/env python3
"""Figure 5.1 - Detection precision by expected phoneme.
 
Usage:
    python scripts/make_figure_5_1.py \
        --input results/so762_eval/knowledge_precision.json \
        --out figures/figure_5_1.png
 
If the key names below do not match your JSON, adjust FIELD_* constants.
Run with --dump-keys to print the keys actually present.
"""
 
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
 
# --- adjust these if your JSON uses different names -----------------------
KEY_TABLE = "precision_by_expected_phoneme"
FIELD_PHONEME = "expected"
FIELD_FLAGGED = "flagged"
FIELD_PRECISION = "precision"
FIELD_RULE_TARGET = "l1_rule_target"
FIELD_BY_TIER = "by_tier"
TIER_L1 = "l1_specific"
# --------------------------------------------------------------------------
 
MIN_FLAGGED = 30          # same threshold as the analysis script
CORPUS_PRECISION = 0.069  # Table 5.1
BASE_RATE = 0.024         # Table 5.1
 
COL_OTHER = "#B8BDC4"     # non-rule-target bars
COL_RULE = "#2E4057"      # rule-target bars
COL_MARK = "#C0392B"      # rule-matched direction marker
 
 
def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if KEY_TABLE not in data:
        raise SystemExit(
            f"key {KEY_TABLE!r} not found. top-level keys: {list(data)}"
        )
    return data[KEY_TABLE]
 
 
def rule_matched_precision(row):
    """Precision of the tier-1 (rule-matched) subset, or None."""
    tiers = row.get(FIELD_BY_TIER) or {}
    t1 = tiers.get(TIER_L1) or {}
    n = t1.get(FIELD_FLAGGED) or 0
    if not n:
        return None
    tr = t1.get("TR", t1.get("tr", 0))
    return tr / n, n
 
 
def build(rows):
    kept = [
        r for r in rows
        if (r.get(FIELD_FLAGGED) or 0) >= MIN_FLAGGED
        and r.get(FIELD_PRECISION) is not None
    ]
    kept.sort(key=lambda r: r[FIELD_PRECISION])  # ascending -> highest on top
    return kept
 
 
def render(rows, out: Path):
    labels, values, colours = [], [], []
    marks = []  # (y_index, precision, n)
 
    for i, r in enumerate(rows):
        n = r[FIELD_FLAGGED]
        labels.append(f"/{r[FIELD_PHONEME]}/  (n={n})")
        values.append(r[FIELD_PRECISION])
        # colour only phonemes where a rule actually matched at least one
        # flagged case; si_palatalization targets /s/ but matched none.
        rm = rule_matched_precision(r) if r.get(FIELD_RULE_TARGET) else None
        colours.append(COL_RULE if rm else COL_OTHER)
        if rm:
            marks.append((i, rm[0], rm[1]))
 
    height = max(3.0, 0.20 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, height))
 
    ax.barh(range(len(rows)), values, color=colours, height=0.68, zorder=3)
 
    for y, p, n in marks:
        ax.plot([p, values[y]], [y, y], color=COL_MARK,
                lw=0.9, ls=":", zorder=4)
        ax.plot(p, y, marker="o", ms=6, mfc="white",
                mec=COL_MARK, mew=1.6, zorder=5)
        ax.annotate(f"n={n}", (p, y), textcoords="offset points",
                    xytext=(6, 4), ha="left", va="bottom",
                    fontsize=7, color=COL_MARK)
 
    ax.axvline(CORPUS_PRECISION, color="#444444", lw=1.0, ls="--", zorder=2)
    ax.axvline(BASE_RATE, color="#888888", lw=1.0, ls=":", zorder=2)
 
    foot = -0.62
    ax.annotate("corpus precision 0.069", (CORPUS_PRECISION, foot),
                textcoords="offset points", xytext=(4, 0),
                fontsize=7.5, color="#444444", va="center")
    ax.annotate("base rate 0.024", (BASE_RATE, foot),
                textcoords="offset points", xytext=(-4, 0),
                fontsize=7.5, color="#888888", va="center", ha="right")
 
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Precision of flagged detections", fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_ylim(-1.05, len(rows) - 0.25)
    ax.tick_params(axis="x", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#E4E7EA", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
 
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COL_RULE),
        plt.Rectangle((0, 0), 1, 1, fc=COL_OTHER),
        Line2D([], [], marker="o", ls="none", ms=6,
               mfc="white", mec=COL_MARK, mew=1.6),
    ]
    ax.legend(
        handles,
        ["targeted by an L1 rule", "not targeted",
         "rule-matched direction only"],
        loc="lower right", fontsize=7.5, frameon=False,
    )
 
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    print(f"wrote {out} and {out.with_suffix('.svg')} ({len(rows)} phonemes)")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("results/so762_eval/knowledge_precision.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/figure_5_1.png"))
    ap.add_argument("--dump-keys", action="store_true")
    a = ap.parse_args()
 
    if a.dump_keys:
        data = json.loads(a.input.read_text(encoding="utf-8"))
        print("top-level:", list(data))
        rows = data.get(KEY_TABLE) or []
        if rows:
            print("row keys:", list(rows[0]))
        return
 
    render(build(load_rows(a.input)), a.out)
 
 
if __name__ == "__main__":
    main()