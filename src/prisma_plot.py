"""Draw the PRISMA 2020 flow diagram from prisma_counts.json.
Usage: python prisma_plot.py prisma_counts.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc="white", ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#333", lw=1))


def main(src, out):
    c = json.load(open(src))
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
    L, W, R, RW, H = 0.3, 5.0, 6.0, 3.8, 1.2
    ys = [9.4, 7.4, 5.4, 3.4, 1.0]
    box(ax, L, ys[0], W, H, f"Records identified from PubMed\nSearch 1: n = {c['identified_search1']}\n"
        f"Search 2 (broadened): n = {c['identified_search2']}")
    box(ax, R, ys[0], RW, H, f"Duplicates removed\n(search 1 fully contained in search 2)\nn = {c['duplicates_removed']}")
    box(ax, L, ys[1], W, H, f"Records after de-duplication\nn = {c['records_after_dedup']}")
    box(ax, R, ys[1], RW, H, f"Excluded by publication type\n(review, editorial, comment, ...)\nn = {c['excluded_pubtype']}")
    box(ax, L, ys[2], W, H, f"Titles/abstracts screened\nn = {c['screened_title_abstract']}")
    box(ax, R, ys[2], RW, H, f"Excluded by keyword rule\n(no subgrouping + brain-molecular terms,\nor animal only) n = {c['excluded_rule']}")
    box(ax, L, ys[3], W, H, f"Full title/abstract review\nn = {c['manual_review']}\n(rule-kept {c['manual_review_from_rule_kept']}"
        f" + title keyword check {c['manual_review_from_title_check']})")
    box(ax, R, ys[3], RW, H, f"Excluded on review\nn = {c['excluded_manual']}\n(single-cell, blood, imaging,\n"
        "epigenome-only, diagnosis-only, ...)")
    box(ax, L, ys[4], W, H * 1.3, f"Included: {c['included_records']} records = {c['included_studies']} studies\n"
        f"({c['included_de_novo']} de novo subtyping, {c['included_application']} applications)\n"
        f"+ 1 study from other sources (Lai 2022)\nTotal {c['included_studies_total']} studies")
    for i in range(4):
        arrow(ax, L + W / 2, ys[i], L + W / 2, ys[i + 1] + (H if i < 3 else H * 1.3))
        arrow(ax, L + W, ys[i] + H / 2, R, ys[i] + H / 2)
    ax.text(5, 10.85, "Literature screening of AD brain transcriptomic subtype studies", ha="center", fontsize=11, weight="bold")
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:3])
