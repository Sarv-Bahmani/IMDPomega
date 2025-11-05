import statistics
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, FrozenSet, List
import csv
from pathlib import Path

import sys
sys.setrecursionlimit(200000)


csv_path = Path("gen_imdp_info/IMDPs_info.csv")
root_models = Path("MDPs")

State = int
QState = int
Action = str
ProdState = Tuple[State, QState]
Label = FrozenSet[str]


iter_period = 2


from imdp import IMDP
from automata import Automata
from product import Product


def expectation_for_action(intervals_list: List[Tuple[ProdState, float, float]], V: Dict[ProdState, float], alpha=1) -> float:
    base = 0.0
    residual = 1.0
    items: List[Tuple[ProdState, float, float, float]] = []  # (y,l,u,V[y])
    for y, l, u in intervals_list:
        base += l * V.get(y, 0.0)
        residual -= l
        items.append((y, l, u, V.get(y, 0.0)))
    items.sort(key=lambda t: t[3])  # ascending V
    exp = base
    r = max(0.0, residual)
    for y, l, u, vy in items:
        if r <= 0: 
            break
        add = min(u - l, r)
        exp += add * vy
        r -= add
    exp = alpha * exp
    return exp

def calc_init_mean(P, L, U):
    mean_i_L = []
    mean_i_U = []
    for (s, q) in P.init_states:
        mean_i_L.append(L[(s, q)])
        mean_i_U.append(U[(s, q)])
    # mean_L = sum(mean_i_L) / len(mean_i_L) if mean_i_L else 0.0
    # mean_U = sum(mean_i_U) / len(mean_i_U) if mean_i_U else 0.0
    mean_L = statistics.mean(mean_i_L) if mean_i_L else 0.0
    mean_U = statistics.mean(mean_i_U) if mean_i_U else 0.0
    return mean_L, mean_U

def interval_iteration(P, eps, max_iter = 151):
    L: Dict[ProdState, float] = {x: 0.0 for x in P.states}
    U: Dict[ProdState, float] = {x: 1.0 for x in P.states}

    L.update({x: 1.0 for x in P.target})
    U.update({x: 0.0 for x in P.losing_sink})   

    mean_L_list, mean_U_list = [], []

    for iterator in range(max_iter):

        if iterator % iter_period == 0 and iterator > 0:            
            if iterator % 10 == 0:
                print("Iteration:", iterator)
            mean_L, mean_U = calc_init_mean(P, L, U)
            mean_L_list.append(mean_L)
            mean_U_list.append(mean_U)

        deltaL = 0.0
        deltaU = 0.0

        for x in P.states:
            if x in P.target:
                continue
            if x in P.losing_sink:
                continue
            acts = P.actions.get(x, ())
            if not acts:
                newL = 0.0
                newU = 0.0
            else:
                best_min = 0
                best_max = 0
                for a in acts:
                    iv = P.trans_prod.get((x, a), {})
                    if not iv:
                        continue
                    iv_list = [(y, l, u) for y, (l, u) in iv.items()]
                    mexp = expectation_for_action(iv_list, L)
                    Mexp = expectation_for_action(iv_list, U, alpha=0.999)
                    best_min = max(best_min, mexp)
                    best_max = max(best_max, Mexp)
                newL = best_min
                newU = best_max

            deltaL = max(deltaL, abs(newL - L[x]))
            deltaU = max(deltaU, abs(newU - U[x]))
            L[x], U[x] = newL, newU

        # gap = max(U[x] - L[x] for x in P.states)
        # if max(deltaL, deltaU) <= eps and gap <= eps:
        if all(U[x] <= L[x] for x in P.states):
            print("breakkkkkk Converged at iteration", iterator)
            break

    return L, U, iterator, mean_L_list, mean_U_list

def value_iteration_scope(P, eps):
    start_time = time.perf_counter()
    L, U, iterator, mean_L_list, mean_U_list  = interval_iteration(P, eps=eps)
    execution_time = time.perf_counter() - start_time
    return {
        "mean_L_list": mean_L_list,
        "mean_U_list": mean_U_list,
        "L": L,   
        "U": U,
        "Convergence_iteration": iterator,
        "Execution_time_sec": execution_time
    }



def constants_vs_var(adds, variable):
    results = []
    for add in adds:
        with csv_path.open(newline='', encoding="utf-8") as f:
            rows = csv.DictReader(f)
            for row in rows:
                if row["address"].strip() == add:
                    variable_val = float(row[variable])
                    results.append({variable: variable_val, "Execution_time_sec": float(row["Execution_time_sec"])})

def plot_x(results, x_var, y_var, pic_name, x_lab, unit=1):
    results.sort(key=lambda d: d[x_var])
    xs = [d[x_var]/unit for d in results]
    ys = [d[y_var] for d in results]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_lab)
    plt.grid(True)
    plt.savefig(f"{pic_name}.png")

def update_csv_reslt(csv_path, address, res):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    row_found = False
    for row in rows:
        if row["address"].strip() == address:
            row["Execution_time_sec"] = f"{res['Execution_time_sec']:.6f}"
            row["Convergence_iteration"] = str(res["Convergence_iteration"])
            # row["Qualitative_time_sec"] = str(res.get("Qualitative_time_sec", ""))
            row_found = True
            break

    if not row_found:
        row = {fn: "" for fn in fieldnames}
        row["address"] = address
        row["Execution_time_sec"] = f"{res['Execution_time_sec']:.6f}"
        row["Convergence_iteration"] = str(res["Convergence_iteration"])
        # row["Qualitative_time_sec"] = str(res.get("Qualitative_time_sec", ""))
        row["Noise Samples"] = str(20000)
        rows.append(row)

    with csv_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_imdp(address, noise_samples, eps=1e-9):

    I = IMDP(address=address, noise_samples=noise_samples)

    all_labsets = {I.label[s] for s in I.states}
    B = Automata(all_labsets, "my_automaton.hoa", read_from_hoa=False)
    P = Product(I, B)
    


    # print('product is build')

    print("Number of product states:", len(P.states))
    common_init_target = P.init_states & P.target
    common_init_losing = P.init_states & P.losing_sink
    # common_target_losing = P.target & P.losing_sink

    print("Init ∩ Target:", len(common_init_target))
    print("Init ∩ Losing:", len(common_init_losing))
    # print("Target ∩ Losing:", len(common_target_losing))

    only_init = P.init_states - (P.target | P.losing_sink)
    print("only init:", len(only_init))

    all_inits = len(P.init_states)
    print("all inits:", all_inits)

    res = value_iteration_scope(P, eps)
    res.update({"Qualitative_time_sec": P.qualitative_time_sec})
    update_csv_reslt(csv_path, address, res)
    return res


adds = [
'Ab_UAV_10-16-2025_20-48-14',
'Ab_UAV_10-16-2025_13-57-21',
'Ab_UAV_10-16-2025_15-11-36',
'Ab_UAV_10-16-2025_15-16-07',
'Ab_UAV_10-16-2025_15-25-59',
'Ab_UAV_10-16-2025_15-29-37'
]
# for add in adds:

add = adds[0]
res = run_imdp(address=add, noise_samples=20000, eps=1e-9)




mean_L_list = res["mean_L_list"]
mean_U_list = res["mean_U_list"]

x_values = list(range(iter_period, (len(mean_L_list)+1) * iter_period, iter_period))

plt.plot(x_values, mean_L_list, marker='o', label='Mean Lower bound')
plt.plot(x_values, mean_U_list, marker='s', label='Mean Upper bound')

plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"Evolution_MeanL_MeanU_InitSt_VI_{add}.png")


