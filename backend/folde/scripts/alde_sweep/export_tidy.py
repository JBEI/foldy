"""Export a tidy per-round dataset from model_evals checkpoints.

One row per (benchmark, method, dms_id, simulation, round) with the cumulative metrics the
paper and the reviewer responses use. This is the grain follow-on analysis wants: Wilcoxon
tests across datasets, regret / trajectory-AUC, and per-target breakdowns all aggregate up
from here, and nothing has to re-parse the 10-50 MB checkpoint JSONs again.

Emits however many rounds each run actually has, so it works before and after the 6-round
ALDE re-run. Check the `n_rounds` column before comparing methods at a given round.

Usage:  python folde/scripts/alde_sweep/export_tidy.py <benchmark> <out.csv>
        benchmark: "single" or "multi"
"""

import csv
import json
import os
import sys

MODEL_EVALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "model_evals")

SOURCES = {
    "single": {
        "Random": "251003-test-benchmark_Random.json",
        "RandomForest": "251003-test-benchmark_RandomToRandomForest-base.json",
        "ZeroShot": "251003-test-benchmark_OnlyNaturalness.json",
        "FolDE": "251003-test-benchmark_FolDE-base.json",
        "ALDE": "260920-alde-fig1_ALDE.json",
        "ALDEzs": "260920-alde-fig1_ALDEzs.json",
    },
    "multi": {
        "Random": "251003-multimutant-benchmark_Random.json",
        "RandomForest": "251003-multimutant-benchmark_RandomToRandomForest-base.json",
        "ZeroShot": "251003-multimutant-benchmark_OnlyNaturalness.json",
        "FolDE": "251003-multimutant-benchmark_FolDE-base.json",
        "ALDE": "260920-alde-fig1-multi_ALDE.json",
        "ALDEzs": "260920-alde-fig1-multi_ALDEzs.json",
    },
}

FIELDS = [
    "benchmark",
    "method",
    "dms_id",
    "sim",
    "round",
    "measured_mutants",
    "cum_top10_count",
    "found_top1",
    "best_percentile",
    "best_norm_activity",
    "pool_size",
    "n_rounds",
    "model_spearman",
]


def rows_for(benchmark, method, path):
    with open(path) as fh:
        d = json.load(fh)
    for camp in d["campaign_results"]:
        lo, hi = camp["min_activity"], camp["max_activity"]
        rng = (hi - lo) if hi > lo else 1.0
        for cfg in camp["config_results"]:
            for sim_idx, s in enumerate(cfg["simulation_results"]):
                n_rounds = max((m["round_found"] for m in s["mutant_metrics"]), default=0)
                spearman = {rm["round_num"]: rm.get("model_spearman") for rm in s["round_metrics"]}
                for r in range(1, n_rounds + 1):
                    seen = [m for m in s["mutant_metrics"] if m["round_found"] <= r]
                    pcts = [m["percentile"] for m in seen]
                    acts = [m["activity"] for m in seen]
                    yield {
                        "benchmark": benchmark,
                        "method": method,
                        "dms_id": camp["dms_id"],
                        "sim": sim_idx,
                        "round": r,
                        "measured_mutants": r * camp["round_size"],
                        "cum_top10_count": sum(1 for p in pcts if p >= 0.90),
                        "found_top1": int(any(p >= 0.99 for p in pcts)),
                        "best_percentile": round(max(pcts), 6) if pcts else "",
                        "best_norm_activity": round((max(acts) - lo) / rng, 6) if acts else "",
                        "pool_size": s["variant_pool_size"],
                        "n_rounds": n_rounds,
                        "model_spearman": spearman.get(r, ""),
                    }


def main():
    benchmark, out_path = sys.argv[1], sys.argv[2]
    n = 0
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for method, fname in SOURCES[benchmark].items():
            path = os.path.join(MODEL_EVALS, fname)
            if not os.path.exists(path):
                print(f"  !! missing {fname}", file=sys.stderr)
                continue
            c = 0
            for row in rows_for(benchmark, method, path):
                w.writerow(row)
                c += 1
                n += 1
            print(f"  {method:13} {c:6d} rows", file=sys.stderr)
    print(f"wrote {out_path} ({n} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
