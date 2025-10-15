from dataclasses import dataclass, asdict
from pathlib import Path
import csv, json, datetime, platform, os, sys

COLUMNS = [
    "address",
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





address = 'Ab_UAV_10-14-2025_16-35-35'
run = {
    "address": address,
    "PRISM Path": "/home/sarv/SarvWork/prism/prism/prism/bin/prism",
    "Model File": "JAIR22_models",
    "Model Name": "UAV",
    "timebound": 32,
    "Monte Carlo Iter": 1000,
    "x_init": [],
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
    "UAV_dim": 3,
    "noise_factor": 1,
    "Regions (base)": 25515,
    "Exported States (PRISM)": 25518,
    "Choices": 1211426,
    "Transitions": 108323004,
    "Noise Factor": 1,
    "Partition": [7, 4, 7, 4, 7, 4],
    "Enabled (total)": 25182,
    "Enabled (init)": "",
    "Deadlocks": "",
    "PRISM Ver": "4.8.1",
    "Property": 'Pmaxmin=? [ F<=16 "reached" ]',
    "PRISM Iter": "",
    "Range (init states)": "",
    "Final Result": "",
    "DefAct (s)": 66.770842313766,
    "ProbCalc (s)": 144.51055264473,
    "Export (s)": 465.586446762085,
    "Build (s)": "",
    "Check (s)": "",
    "Total (s)": "",
    "MC Init": "",
    "MC Summary": "",
    "Warnings": "Unknown args: --tau 2.0; switched to explicit engine; Java OOM (heap) during build; no PRISM vector/policy exported.",
    **_default_env(),
    "Execution_time_sec": "",
    "Convergence_iteration": ""
}


csv_path = Path("gen_imdp_info/IMDPs_info.csv")
append_row(csv_path, run)


# csv_path = Path("gen_imdp_info/IMDPs_info.csv")

# # read the existing rows
# with csv_path.open(newline='', encoding='utf-8') as f:
#     reader = csv.DictReader(f)
#     rows = list(reader)
#     fieldnames = reader.fieldnames or []
#
# # add the new columns if not already there
# if "Execution_time_sec" not in fieldnames:
#     fieldnames.append("Execution_time_sec")
# if "Convergence_iteration" not in fieldnames:
#     fieldnames.append("Convergence_iteration")

# # fill the new columns with blanks (or default values)
# for row in rows:
#     if "Execution_time_sec" not in row or not row["Execution_time_sec"]:
#         row["Execution_time_sec"] = ""
#     if "Convergence_iteration" not in row or not row["Convergence_iteration"]:
#         row["Convergence_iteration"] = ""

# overwrite the same CSV with the new columns
# with csv_path.open("w", newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)



