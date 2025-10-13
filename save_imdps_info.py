from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json, datetime, platform, os, sys

COLUMNS = [
    "address","PRISM Path",
    "Model File","Model Name","timebound",
    "Monte Carlo Iter","x_init","Noise Samples","Confidence",
    "Sample Clustering","Iterations",
    "drone_mc_step","drone_mc_iter",
    "bld_partition","bld_target_size","bld_par_uncertainty",
    "drug_partition","UAV_dim","noise_factor",
    "Regions (base)","Exported States (PRISM)","Choices","Transitions",
    "Noise Factor",
    "Partition","Enabled (total)","Enabled (init)","Deadlocks",
    "PRISM Ver",
    "Property","PRISM Iter","Range (init states)","Final Result",
    "DefAct (s)","ProbCalc (s)","Export (s)","Build (s)","Check (s)","Total (s)",
    "MC Init","MC Summary","Warnings","Python Ver","OS","Conda Env"
]



def _default_env():
    return {
        "Python Ver": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "OS": platform.system(),
        "Conda Env": os.environ.get("CONDA_DEFAULT_ENV",""),
    }

def ensure_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

def append_row(csv_path: Path, row: dict):
    ensure_csv(csv_path)
    # keep only known columns; missing keys become ""
    clean = {k: row.get(k, "") for k in COLUMNS}
    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(clean)


# if __name__ == "__main__":
address = 'Ab_UAV_10-13-2025_21-20-12'
run = {
    "address": address,
    "PRISM Path": "/home/sarv/SarvWork/prism/prism/prism/bin/prism",
    "Model File": "JAIR22_models",
    "Model Name": "UAV",
    "timebound": 8,
    "Monte Carlo Iter": 1000,
    "x_init": [-6, 0, -6, 0],
    "Noise Samples": 3200,
    "Confidence": 1e-08,
    "Sample Clustering": 0.01,
    "Iterations": 1,
    "drone_mc_step": 0.2,
    "drone_mc_iter": 100,
    "bld_partition": [25, 35],
    "bld_target_size": [[-0.1, 0.1], [-0.3, 0.3]],
    "bld_par_uncertainty": False,
    "drug_partition": [20, 20, 20],
    "UAV_dim": 2,
    "noise_factor": 1,
    "Regions (base)": 784,
    "Exported States (PRISM)": 787,
    "Choices": 11388,
    "Transitions": 417998,
    "Noise Factor": 1,
    "Partition": [7, 4, 7, 4],
    "Enabled (total)": 667,
    "Enabled (init)": 25,
    "Deadlocks": 128,
    "PRISM Ver": "4.8.1",
    "Property": 'Pmaxmin=? [ F<=4 "reached" ]',
    "PRISM Iter": 4,
    "Range (init states)": [0.0, 1.0],
    "Final Result": [0.0, 1.0],
    "DefAct (s)": 0.09338760376,
    "ProbCalc (s)": 2.889494657516,
    "Export (s)": 2.376542806625,
    "Build (s)": 1.809,
    "Check (s)": 0.543,
    "Total (s)": 5.05219078064,
    "MC Init": 58,
    "MC Summary": "MC 1 run (same init)",
    "Warnings": "Switched to explicit engine; deadlocks fixed in 128 states; total 2 warnings",
    **_default_env(),
}


csv_path = Path("gen_imdp_info/IMDPs_info.csv")
append_row(csv_path, run)






