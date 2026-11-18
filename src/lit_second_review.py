"""Second-reviewer check of title/abstract screening.

make:  python lit_second_review.py make records_prescreened.csv screening_decisions.csv out_dir
score: python lit_second_review.py score out_dir/key.csv filled_sheet.xlsx out_dir

The sheet is blind: it does not show the first reviewer's decision. A stratified random
sample (included / excluded on review / excluded by rule) is shuffled together.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import os
N = {"included": 10, "excluded_manual": 25, "excluded_rule": 25}
SEED = 20261117
if os.environ.get("ROUND") == "2":  # human-only round, disjoint from round 1
    N = {"included": 6, "excluded_manual": 12, "excluded_rule": 12}
    SEED = 20261118


def make(records, decisions, out):
    out = Path(out)
    r = pd.read_csv(records, dtype={"PMID": str})
    d = pd.read_csv(decisions, dtype={"PMID": str})
    d = d.merge(r[["PMID", "abstract"]], on="PMID")
    prev = Path(os.environ["EXCLUDE_KEY"]) if os.environ.get("EXCLUDE_KEY") else None
    if prev is not None:
        d = d[~d.PMID.isin(pd.read_csv(prev, dtype={"PMID": str}).PMID)]
    parts = [d[d.stage == s].sample(n, random_state=SEED) for s, n in N.items()]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s.insert(0, "no", range(1, len(s) + 1))
    s[["no", "PMID", "stage", "reason"]].to_csv(out / "key.csv", index=False)
    sheet = s[["no", "PMID", "year", "title", "abstract"]].copy()
    sheet["판정(포함/제외)"] = ""
    sheet["메모"] = ""
    with pd.ExcelWriter(out / os.environ.get("SHEET", "2차검토_시트.xlsx"), engine="openpyxl") as w:
        sheet.to_excel(w, index=False, sheet_name="검토")
        ws = w.sheets["검토"]
        from openpyxl.styles import Alignment
        widths = {"A": 5, "B": 10, "C": 6, "D": 45, "E": 90, "F": 16, "G": 25}
        for c, wd in widths.items():
            ws.column_dimensions[c].width = wd
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"  # freezing A-E hid the decision column on narrow screens
    print(len(s), "records written")


def score(key, filled, out):
    k = pd.read_csv(key, dtype={"PMID": str})
    f = pd.read_excel(filled, sheet_name="검토", dtype={"PMID": str})
    col = [c for c in f.columns if str(c).startswith("판정")][0]
    m = k.merge(f[["no", col, "메모"]], on="no")
    m["r2"] = m[col].astype(str).str.strip().map(lambda x: "include" if x.startswith("포함") else
                                                 ("exclude" if x.startswith("제외") else None))
    m = m[m.r2.notna()].copy()
    m["r1"] = (m.stage == "included").map({True: "include", False: "exclude"})
    po = (m.r1 == m.r2).mean()
    p1, p2 = (m.r1 == "include").mean(), (m.r2 == "include").mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    dis = m[m.r1 != m.r2]
    res = {"n_rated": int(len(m)), "agreement": round(po, 3), "cohen_kappa": round(kappa, 3),
           "by_stratum": m.groupby("stage").apply(lambda g: round((g.r1 == g.r2).mean(), 3)).to_dict(),
           "disagreements": dis[["no", "PMID", "stage", "reason", "r2", "메모"]].to_dict("records")}
    Path(out, "agreement.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]](*sys.argv[2:])
