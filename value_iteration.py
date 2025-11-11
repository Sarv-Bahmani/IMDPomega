import statistics
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, FrozenSet, List
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


iter_init_save = 1
iter_print = 5
up_contrac_fctr = 0.999

from imdp import IMDP
from automata import Automata
from product import Product


address_str = "address"
val_iter_time_str = "Val_Iter_Execution_time_sec"
val_iter_converge_iter_str = "Val_Iter_Convergence_iteration"
qual_time_str = "Qualitative_time_sec"
transitions_str = 'Transitions'
Exported_States_PRISM_str = 'Exported States (PRISM)'


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

def interval_iteration(P, eps, max_iter = 51):
    L: Dict[ProdState, float] = {x: 0.0 for x in P.states}
    U: Dict[ProdState, float] = {x: 1.0 for x in P.states}

    L.update({x: 1.0 for x in P.target})
    U.update({x: 0.0 for x in P.losing_sink})   

    mean_L_list, mean_U_list = [], []

    for iterator in range(max_iter):

        if iterator % iter_init_save == 0 and iterator > 1:            
            if iterator % iter_print == 0:
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
                    Mexp = expectation_for_action(iv_list, U, alpha=up_contrac_fctr)
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
            print("VALUE ITERATION breakkkkkk Converged at iteration", iterator)
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
        val_iter_converge_iter_str: iterator,
        val_iter_time_str: execution_time
    }

def plot_init_evolution_val_iter(res, add):
    mean_L_list = res["mean_L_list"]
    mean_U_list = res["mean_U_list"]

    x_values = list(range(iter_init_save, (len(mean_L_list)+1) * iter_init_save, iter_init_save))

    plt.plot(x_values, mean_L_list, marker='o', label='Mean Lower bound')
    plt.plot(x_values, mean_U_list, marker='s', label='Mean Upper bound')

    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Evolution_InitSt_VI_{add}.png")
    plt.close()










# def constants_vs_var(adds, variable):
#     results = []
#     for add in adds:
#         with csv_path.open(newline='', encoding="utf-8") as f:
#             rows = csv.DictReader(f)
#             for row in rows:
#                 if row["address"].strip() == add:
#                     variable_val = float(row[variable])
#                     results.append({variable: variable_val, "Execution_time_sec": float(row["Execution_time_sec"])})









