"""Build main-text Table 1: method comparison at round 3, pooled over 20 targets."""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

ME = os.path.join(os.path.dirname(__file__), "..", "model_evals")

SINGLE = [
    "A4GRB6_PSEAI_Chen_2020",
    "AMIE_PSEAE_Wrenbeck_2017",
    "CAS9_STRP1_Spencer_2017_positive",
    "HMDH_HUMAN_Jiang_2019",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "KKA2_KLEPN_Melnikov_2014",
    "LGK_LIPST_Klesmith_2015",
    "MET_HUMAN_Estevam_2023",
    "MLAC_ECOLI_MacRae_2023",
    "MSH2_HUMAN_Jia_2020",
    "MTHR_HUMAN_Weile_2021",
    "PAI1_HUMAN_Huttinger_2021",
    "PPARG_HUMAN_Majithia_2016",
    "PTEN_HUMAN_Mighell_2018",
    "RNC_ECOLI_Weeks_2023",
    "S22A1_HUMAN_Yee_2023_activity",
    "SC6A4_HUMAN_Young_2021",
]
MULTI = ["SPG1_STRSG_Olson_2014", "GRB2_HUMAN_Faure_2021", "PABP_YEAST_Melamed_2013"]
ALL20 = SINGLE + MULTI

ROUND = 3

# label -> list of (file glob, config name)
SOURCES = {
    "FolDE": [
        ("251003-test-benchmark_FolDE-base.json", "FolDE-base"),
        ("251003-multimutant-benchmark_FolDE-base.json", "FolDE-base"),
    ],
    "Zero-shot": [
        ("251003-test-benchmark_OnlyNaturalness.json", "OnlyNaturalness"),
        ("251003-multimutant-benchmark_OnlyNaturalness.json", "OnlyNaturalness"),
    ],
    "Random": [
        ("251003-test-benchmark_Random.json", "Random"),
        ("251003-multimutant-benchmark_Random.json", "Random"),
    ],
    "RF-OHE": [
        ("251003-test-benchmark_RandomToRandomForest-base.json", "RandomToRandomForest-base"),
        (
            "251003-multimutant-benchmark_RandomToRandomForest-base.json",
            "RandomToRandomForest-base",
        ),
    ],
    "RF-ESM": [
        (
            "251003-test-benchmark_RandomToRandomForest-300m-embeddings.json",
            "RandomToRandomForest-300m-embeddings",
        ),
        (
            "251003-multimutant-benchmark_RandomToRandomForest-300m-embeddings.json",
            "RandomToRandomForest-300m-embeddings",
        ),
    ],
    "ALDE": [("260920-alde-fig1_ALDE.json", "ALDE"), ("260920-alde-fig1-multi_ALDE.json", "ALDE")],
    "ALDEzs": [
        ("260920-alde-fig1_ALDEzs.json", "ALDEzs"),
        ("260920-alde-fig1-multi_ALDEzs.json", "ALDEzs"),
    ],
    "Ridge-ESM": [("261001-ridge_Ridge-ESM.json", "Ridge-ESM")],
    "Ridge-ESM-aug": [("261002-ridge-aug_Ridge-ESM-aug.json", "Ridge-ESM-aug")],
    "Ridge-OHE-aug": [("261002-ridge-aug_Ridge-OHE-aug.json", "Ridge-OHE-aug")],
}


def load(label):
    """-> {dms_id: {'top10': mean cum top-10% count @ROUND,
    'p1': frac of sims that found a top-1% mutant by ROUND}}"""
    out = {}
    for fn, cfgname in SOURCES[label]:
        path = os.path.join(ME, fn)
        d = json.load(open(path))
        for camp in d["campaign_results"]:
            dms = camp["dms_id"]
            for cr in camp["config_results"]:
                if cr["config"]["name"] != cfgname:
                    continue
                t10, p1 = [], []
                for sim in cr["simulation_results"]:
                    mm = sim["mutant_metrics"]
                    sofar = [m for m in mm if m["round_found"] <= ROUND]
                    t10.append(sum(1 for m in sofar if m["percentile"] >= 0.90))
                    p1.append(1.0 if any(m["percentile"] >= 0.99 for m in sofar) else 0.0)
                if dms in out:
                    raise SystemExit(f"duplicate {label} {dms}")
                out[dms] = {
                    "top10": float(np.mean(t10)),
                    "p1": float(np.mean(p1)),
                    "nsims": len(t10),
                }
    return out


def ratio_stats(folde, other, key):
    """median FolDE/other ratio (0-denominators -> inf, 0/0 -> tie), wins|ties|losses, one-sided p."""
    ratios, a, b = [], [], []
    for dms in ALL20:
        f, o = folde[dms][key], other[dms][key]
        if o == 0 and f == 0:
            ratios.append(1.0)
        elif o == 0:
            ratios.append(np.inf)
        else:
            ratios.append(f / o)
        a.append(f)
        b.append(o)
    a, b = np.array(a), np.array(b)
    d = a - b
    w, t, l = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    if np.any(d != 0):
        p = wilcoxon(a, b, alternative="greater").pvalue
    else:
        p = float("nan")
    return float(np.median(ratios)), (w, t, l), p


def main():
    data = {}
    for label in SOURCES:
        data[label] = load(label)
        missing = [d for d in ALL20 if d not in data[label]]
        extra = [d for d in data[label] if d not in ALL20]
        if missing or extra:
            print(f"!! {label}: missing={missing} extra={extra}", file=sys.stderr)
    folde = data["FolDE"]

    print(
        f"\n{'method':<15} {'top10 mean':>10} {'ratio':>7} {'W|T|L':>9} {'p':>9} "
        f"{'P(top1%)':>9} {'ratio':>7} {'W|T|L':>9} {'p':>9}"
    )
    print("-" * 100)
    f_t10 = np.mean([folde[d]["top10"] for d in ALL20])
    f_p1 = np.mean([folde[d]["p1"] for d in ALL20])
    print(f"{'FolDE':<15} {f_t10:>10.2f} {'ref':>7} {'':>9} {'':>9} {f_p1:>9.3f} {'ref':>7}")
    for label in SOURCES:
        if label == "FolDE":
            continue
        o = data[label]
        m_t10 = np.mean([o[d]["top10"] for d in ALL20])
        m_p1 = np.mean([o[d]["p1"] for d in ALL20])
        r1, (w1, t1, l1), p1v = ratio_stats(folde, o, "top10")
        r2, (w2, t2, l2), p2v = ratio_stats(folde, o, "p1")
        print(
            f"{label:<15} {m_t10:>10.2f} {r1:>7.3f} {w1:>2d}|{t1:>2d}|{l1:>2d} {p1v:>9.4f} "
            f"{m_p1:>9.3f} {r2:>7.3f} {w2:>2d}|{t2:>2d}|{l2:>2d} {p2v:>9.4f}"
        )

    # how many targets where the method never finds a top-1% mutant (mean P == 0)
    print("\nTargets with P(top 1%) == 0 at round 3:")
    for label in SOURCES:
        z = [d for d in ALL20 if data[label][d]["p1"] == 0]
        print(f"  {label:<15} {len(z):>2d}  {z}")


if __name__ == "__main__":
    main()
