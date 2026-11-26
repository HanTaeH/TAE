"""Upper confidence limit with sampling variability, and an equivalence test of classifier vs matched nulls (v12, review round 10, Reviewer 1 M1, M3).

v11 (spike_in_bound.py) inverted the test against the spread of the matched null across gene sets only,
assumed the classifier's offset from the null median to be zero, and returned 0 when the observed
coupling already lay beyond the null's 5th percentile (an empty confidence set, not a limit). Here, from
results/spike_in_real/spike_in_real.json:
  sd_null      = (null median - null q05) / 1.645   (spread over gene sets, same samples);
  sd_boot(f)   = (rho_median_f - rho_q05_f) / 1.645 (spread over bootstrap resamples of AD samples and the
                 random subtype signal, at spike share f);
  H_f: rho_classifier ~ N(null median + delta_rho(f), sd_null^2 + sd_boot(f)^2), rejected (one-sided 5%)
       when rho_obs < null median + delta_rho(f) - 1.645 sd.
  (a) UCL, zero offset: largest f not rejected (linear interpolation; "empty" if f = 0 is rejected,
      "beyond grid" if the largest f is not rejected), with the share of C-A variance it explains;
  (b) offset-free detectable shift: smallest f whose shift exceeds 1.645 sd, i.e. the signal the
      comparison could detect whatever the classifier's own offset; it does not depend on rho_obs.
  Equivalence (TOST): per cohort offset d = rho_obs - null median, SE = sqrt(sd_null^2 + sd_boot(0)^2);
  inverse-variance pooled offset with its 90% CI, equivalent within +/-0.05 (and +/-0.03) if the CI lies
  inside the margin. Rows for the specificity-, strength(tertile)- and strength(decile)-matched nulls.

Usage: python spike_in_bound_v12.py --res ../results --out ../results/spike_in_v12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

Z = 1.645


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = json.loads((a.res / "spike_in_real/spike_in_real.json").read_text())
    out, eq = {}, {}
    for k in ("spec", "strength_tertile", "strength_decile"):
        offs, ses = [], []
        for c, r in d.items():
            fs = np.array([s["f"] for s in r["spike"]])
            dr = np.maximum.accumulate(np.array([s["delta_rho"] for s in r["spike"]]))
            vs = np.array([s["var_share_median"] for s in r["spike"]])
            sb = np.array([(s["rho_median"] - s["rho_q05"]) / Z for s in r["spike"]])
            nm, q05 = r["nulls"][k]["median"], r["nulls"][k]["q05_q95"][0]
            sn = (nm - q05) / Z
            sd = np.sqrt(sn ** 2 + sb ** 2)
            crit = nm + dr - Z * sd                 # reject H_f when obs < crit
            ok = r["obs_rho"] >= crit
            if not ok[0]:
                ucl, status = None, "empty (f = 0 rejected)"
            elif ok.all():
                ucl, status = float(fs[-1]), "beyond grid"
            else:
                j = int(np.argmin(ok))               # first rejected f
                g = r["obs_rho"] - crit               # positive while not rejected
                ucl = float(fs[j - 1] + (fs[j] - fs[j - 1]) * g[j - 1] / (g[j - 1] - g[j]))
                status = "bounded"
            det = dr - Z * sd
            jj = np.where(det > 0)[0]
            if len(jj) == 0:
                fdet = None
            elif jj[0] == 0:
                fdet = 0.0
            else:
                i = jj[0]
                fdet = float(fs[i - 1] + (fs[i] - fs[i - 1]) * (-det[i - 1]) / (det[i] - det[i - 1]))
            row = out.setdefault(c, {"n_ad": r["n_ad"], "obs_rho": r["obs_rho"]})
            row[k] = {"null_median": nm, "sd_null": round(float(sn), 4), "sd_boot_f0": round(float(sb[0]), 4),
                      "offset": round(r["obs_rho"] - nm, 3), "status": status,
                      "ucl95_f": None if ucl is None else round(ucl, 3),
                      "ucl95_var_share": None if ucl is None else round(float(np.interp(ucl, fs, vs)), 3),
                      "detectable_f": None if fdet is None else round(fdet, 3),
                      "detectable_var_share": None if fdet is None else round(float(np.interp(fdet, fs, vs)), 3)}
            offs.append(r["obs_rho"] - nm)
            ses.append(np.sqrt(sn ** 2 + sb[0] ** 2))
        offs, ses = np.array(offs), np.array(ses)
        w = 1 / ses ** 2
        mu, se = float((w * offs).sum() / w.sum()), float(1 / np.sqrt(w.sum()))
        q = float((w * (offs - mu) ** 2).sum())
        lo, hi = mu - Z * se, mu + Z * se
        eq[k] = {"pooled_offset": round(mu, 3), "se": round(se, 4), "ci90": [round(lo, 3), round(hi, 3)],
                 "cochran_q": round(q, 2), "q_p": round(float(stats.chi2.sf(q, len(offs) - 1)), 4),
                 "equivalent_within_0.05": bool(lo > -0.05 and hi < 0.05),
                 "equivalent_within_0.03": bool(lo > -0.03 and hi < 0.03),
                 "per_cohort_offset": dict(zip(d.keys(), np.round(offs, 3).tolist())),
                 "per_cohort_se": dict(zip(d.keys(), np.round(ses, 4).tolist()))}
    rep = {"bound": out, "equivalence": eq}
    (a.out / "spike_in_bound_v12.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
