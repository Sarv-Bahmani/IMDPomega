# IMDP × Büchi Automaton Experiment Framework

This project reproduces all experiments in our **IMDP × Büchi automaton** paper, using **Value Iteration**, **Policy Iteration**, and reporting the results for **Qualitative Analysis**.
It loads IMDP models, constructs the synchronous product with a Büchi automaton, runs the quantitative solvers, and generates all plots and CSV outputs reported in the paper.

---

## Project Structure

```
IMDPOMEGA/
├── data/
│   └── raw/                      # IMDPs grouped by model type ("shuttle" or "UAV")
│       ├── shuttle/
│       │   ├── 3200_1_Ab_shuttle_...
│       │   ├── 5000_2_Ab_shuttle_...
│       │   └── ...
│       └── UAV/
│           ├── 784-0.5-Ab_UAV_...
│           ├── 1024-1-Ab_UAV_...
│           └── ...
│
├── gen_imdp_info/
│   ├── IMDPs_info_shuttle.csv    # Summary (updated by runner)
│   └── IMDPs_info_uav.csv
│
├── hoa_files/
│   └── my_automaton.hoa          # Büchi automaton (HOA format)
│
├── results/
│   ├── each_imdp_result/         # Per-instance CSV results
│   ├── initial_states/           # VI/PI evolution plots per instance
│   └── plots/                    # All aggregate figures for the paper
│
├── imdp_adds.json                # list of IMDP addresses per model type
│
├── automata.py                   # HOA loader and automaton construction
├── imdp.py                       # IMDP loader (from data/raw directories)
├── product.py                    # IMDP × Automaton product construction
├── value_iteration.py            # Value Iteration implementation
├── strategy_gurobi.py            # Policy Iteration implementation (via Gurobi)
├── imdp_runner.py                # Main experiment driver (entry point)
└── README.md                     # You are here
```

---
## IMDP Address Management

IMDP addresses should be stored in a JSON file:
```json
{
  "shuttle": [
    "3200_1_Ab_shuttle_11-24-2025_17-17-12",
    ...
  ],

  "uav": [
    "784-0.5-Ab_UAV_10-16-2025_20-48-14",
    ...
  ]
}
```

---
## How to Run

Run the full experiment pipeline for a given model type:
```bash
python imdp_runner.py <Model_Type> <Path_to_JSON>
```
### Examples
```bash
python imdp_runner.py uav imdp_adds.json
```

The runner will:

1. Load all IMDPs for the selected model type from `data/raw/<model_type>/<address>/`.
2. Load the Büchi automaton from `hoa_files/my_automaton.hoa`.
3. Construct the synchronous product IMDP × automaton.
4. Perform:

   * **Qualitative analysis**
   * **Value Iteration**
   * **Policy Iteration** (requires Gurobi)
5. Save:

   * VI/PI evolution plots → `results/initial_states/`
   * Per-IMDP CSV results → `results/each_imdp_result/`
   * Global aggregates → updated `IMDPs_info_<type>.csv`
6. Generate all paper figures → `results/plots/`.

---

## Output

For each IMDP instance, the tool:

1. Runs **qualitative analysis**, **value iteration**, and **Policy Iteration**.
2. Records:

   * execution times for both algorithms,
   * Value Iteration convergence iterations,
   * qualitative runtime,
   * PI/VI runtime ratio.
   * evolution of the initial-state value (VI vs PI),

---

## Customization

* Add or remove IMDPs simply by editing `imdp_adds.json`.
* Modify automaton by replacing `hoa_files/my_automaton.hoa`.
* Add/remove IMDPs by placing them under:

  ```
  data/raw/shuttle/
  data/raw/UAV/
  ```

---

## IMDP Input Format
IMDPs are directories.
Each entry in imdp_adds.json must correspond to a subfolder:

```bash
data/raw/shuttle/<address>/
data/raw/UAV/<address>/
```

All required files for an IMDP instance must be inside its folder.

---

## Requirements

* Python **3.10+**
* `numpy`
* `pandas`
* `matplotlib`
* `gurobipy`

---
