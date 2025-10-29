import collections
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from typing import Dict, Set, Tuple, FrozenSet, Iterable, List
import re
import csv
from pathlib import Path


import sys
sys.setrecursionlimit(200000)


sta = "Abstraction_interval.sta"
lab = "Abstraction_interval.lab"
tra = "Abstraction_interval.tra"

csv_path = Path("gen_imdp_info/IMDPs_info.csv")
root_models = Path("MDPs")

State = int
QState = int
Action = str
ProdState = Tuple[State, QState]
Label = FrozenSet[str]

states_str = "states"
init_state_str = "init_state"
actions_str = "actions"
trans_MDP_str = "trans_MDP"
underline = "_"

iter_period = 2


from imdp import IMDP
from automata import Automata


class BuchiA:
    def __init__(self, ap: Set[str]):
        self.ap = set(ap)
        self.Q: Set[QState] = set()
        self.q0 = 0
        self.acc: Set[QState] = set()
        self.trans_automa: Dict[Tuple[QState, Label], Set[QState]] = defaultdict(set)

    def add_state(self, q, initial=False, accepting=False):
        self.Q.add(q)
        if initial: self.q0 = q
        if accepting: self.acc.add(q)

    def add_edge(self, q, lab, q2):
        self.trans_automa[(q, lab)].add(q2)

    def step(self, q, lab) -> Set[QState]:
        return self.trans_automa.get((q, lab), set())




def buchi_reach(all_labsets): 
    B = BuchiA({tok for S in all_labsets for tok in S}) 
    B.add_state(0) 
    B.add_state(1, initial=True, accepting=True) 
    B.add_state(2) 
    for labset in all_labsets: 
        B.add_edge(2, labset, 2) 
        if "failed" in labset: 
            B.add_edge(0, labset, 2) 
            B.add_edge(1, labset, 2) 
        elif "reached" in labset: # and not "deadlock" in labset and not "failed" in labset: 
            B.add_edge(0, labset, 1) 
            B.add_edge(1, labset, 1) 
        else: 
            B.add_edge(0, labset, 0) 
            B.add_edge(1, labset, 1) 
    return B




class Product:
    def __init__(self, imdp, buchi):
        self.imdp = imdp
        self.buchi = buchi
        self.states: Set[ProdState] = set()
        self.init_states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
        self.trans_prod: Dict[Tuple[ProdState, Action], Dict[ProdState, Tuple[float, float]]] = {}
        self.acc_states: Set[ProdState] = set()
        self.graph = defaultdict(set)
        self.build_product()
        self.prod_graph()

        start_time = time.perf_counter()

        self.mecs = self.mec_decomposition()
        self.aecs = self.aecs_from_mecs(self.mecs)
        self.target = set().union(*self.aecs) if self.aecs else set()
        self.win_region = self.almost_sure_winning(self.target)

        self.losing_sink = self.surely_losing()

        execution_time = time.perf_counter() - start_time
        
        self.qualitative_time_sec = execution_time


    def build_product(self):
        # Seed product states: start in (s, q0) for all s
        for s in self.imdp.states:
            for q in self.buchi.Q:
                ps = (s, q)
                self.states.add(ps)
                if q in self.buchi.acc:
                    self.acc_states.add(ps)
                if "init" in self.imdp.label[s]: # and not "failed" in self.imdp.label[s]:
                    if q == self.buchi.q0:
                        self.init_states.add(ps)

        for (s, q) in set(self.states):
            self.trans_update(s, q)
      
    def trans_update(self, s, q):
        for a in self.imdp.actions.get(s, ()):
            outs = self.imdp.intervals.get((s, a), {})
            if not outs: 
                continue
            self.actions[(s, q)].add(a)
            prod_outs = {}
            for s2, (l, u) in outs.items():
                labset = self.imdp.label.get(s, frozenset())
                next_qs = self.buchi.step(q, labset)
                if not next_qs:
                    continue
                for q3 in next_qs:
                    ps = (s2, q3)
                    old = prod_outs.get(ps, (0.0, 0.0))
                    prod_outs[ps] = (old[0] + l, old[1] + u)
                    if q3 in self.buchi.acc:
                        self.acc_states.add(ps)
            self.trans_prod[((s, q), a)] = prod_outs

    def prod_graph(self):
        for (ps, a), outs in self.trans_prod.items():
            for t, (l, u) in outs.items():
                if u > 0:
                    self.graph[ps].add(t)

    def sccs(self):
        nodes = self.states
        edges = self.graph
        idx, low, st, on, comps = {}, {}, [], set(), []
        i = 0
        def dfs(v):
            nonlocal i
            idx[v] = i; low[v] = i; i += 1
            st.append(v); on.add(v)
            for w in edges.get(v, ()):
                if w not in idx:
                    dfs(w); low[v] = min(low[v], low[w])
                elif w in on:
                    low[v] = min(low[v], idx[w])
            if low[v] == idx[v]:
                C = set()
                while True:
                    w = st.pop(); on.remove(w); C.add(w)
                    if w == v: break
                comps.append(C)
        for v in nodes:
            if v not in idx:
                dfs(v)
        return comps

    def closed_actions(self, SCC: Set[ProdState]) -> Dict[ProdState, Set[Action]]:
        keep: Dict[ProdState, Set[Action]] = {}
        for s in SCC:
            kept = set()
            for a in self.actions.get(s, ()):
                outs = self.trans_prod.get((s, a), {})
                if outs and all((t in SCC) for t in outs):
                    kept.add(a)
            if kept:
                keep[s] = kept
        return keep

    def mec_decomposition(self) -> List[Set[ProdState]]:
        mecs: List[Set[ProdState]] = []
        for C in self.sccs():
            SCC = set(C)
            changed = True
            while changed:
                changed = False
                keep = self.closed_actions(SCC)
                drop = [s for s in SCC if s not in keep]
                if drop:
                    for s in drop: SCC.remove(s)
                    changed = True
            if SCC:
                mecs.append(SCC)
        mecs.sort(key=lambda X: -len(X))
        maximal = []
        for i, M1 in enumerate(mecs):
            if any(M1 < M2 for j, M2 in enumerate(mecs) if j != i):
                continue
            maximal.append(M1)
        return maximal

    def aecs_from_mecs(self, mecs: Iterable[Set[ProdState]]) -> List[Set[ProdState]]:
        return [C for C in mecs if any(ps in self.acc_states for ps in C)]

    def almost_sure_winning(self, Targ: Set[ProdState]) -> Set[ProdState]:
        if not Targ:
            return set()
        prod_states = set(self.states)
        changed = True
        while changed:
            changed = False
            keep = self.closed_actions(prod_states)
            to_remove = [state for state in prod_states if state not in keep]
            if to_remove:
                for state in to_remove: prod_states.remove(state)
                changed = True
        Region = set(Targ)
        added = True
        while added:
            added = False
            for statee in list(prod_states):
                if statee in Region:
                    continue
                for a in self.actions.get(statee, ()):
                    outs = self.trans_prod.get((statee, a), {})
                    if outs:
                        if all(t in prod_states for t in outs):
                            if any(t in Region for t in outs):
                                Region.add(statee); added = True; break
        return Region


    def _backward_reachable(self, seeds: Set[ProdState]) -> Set[ProdState]:
        """All states that can reach any seed along edges with u>0 (ignoring actions)."""
        if not seeds:
            return set()
        # Build reverse adjacency on the fly
        rev = defaultdict(set)
        for u, nbrs in self.graph.items():
            for v in nbrs:
                rev[v].add(u)

        seen = set(seeds)
        dq = collections.deque(seeds)
        while dq:
            v = dq.popleft()
            for u in rev.get(v, ()):
                if u not in seen:
                    seen.add(u)
                    dq.append(u)
        return seen

    def surely_losing(self) -> Set[ProdState]:
        """
        States from which NO path can reach any AEC (target) state.
        This is the complement of the backward-reachable set of the AECs.
        """
        # If there are no AECs at all, everything is surely losing.
        if not self.target:
            return set(self.states)
        can_reach_target = self._backward_reachable(self.target)
        return set(self.states) - can_reach_target




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
    mean_L = sum(mean_i_L) / len(mean_i_L) if mean_i_L else 0.0
    mean_U = sum(mean_i_U) / len(mean_i_U) if mean_i_U else 0.0
    return mean_L, mean_U

def interval_iteration(P, eps, max_iter = 51):
    L: Dict[ProdState, float] = {x: 0.0 for x in P.states}
    U: Dict[ProdState, float] = {x: 1.0 for x in P.states}

    L.update({x: 1.0 for x in P.target})
    U.update({x: 0.0 for x in P.losing_sink})   

    mean_L_list, mean_U_list = [], []

    for iterator in range(max_iter):

        if iterator % iter_period == 0 and iterator > 0:            
            if iterator == 10:
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
                best_min = None
                best_max = None
                for a in acts:
                    iv = P.trans_prod.get((x, a), {})
                    if not iv:
                        continue
                    iv_list = [(y, l, u) for y, (l, u) in iv.items()]
                    mexp = expectation_for_action(iv_list, L)
                    Mexp = expectation_for_action(iv_list, U, alpha=0.999)
                    best_min = mexp if best_min is None else max(best_min, mexp)
                    best_max = Mexp if best_max is None else max(best_max, Mexp)
                newL = best_min if best_min is not None else 0.0
                newU = best_max if best_max is not None else 0.0

            deltaL = max(deltaL, abs(newL - L[x]))
            deltaU = max(deltaU, abs(newU - U[x]))
            L[x], U[x] = newL, newU

        # gap = max(U[x] - L[x] for x in P.states)
        # if max(deltaL, deltaU) <= eps and gap <= eps:

        if all(U[x] <= L[x] for x in P.states):
            print("breakkkkkk Converged at iteration", iterator)
            break

    return L, U, iterator, mean_L_list, mean_U_list

def quantitative_buchi_imdp(P, eps):
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

def row_already_calced(csv_path, address):
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["address"].strip() == address and row["Execution_time_sec"] != "":
                return True
    return False




def run_imdp(address, noise_samples, eps=1e-9):
    # is_row_already_calced = row_already_calced(csv_path, address)
    # if is_row_already_calced:
    #     print(address)
    #     print("^ already calced")
        # return

    I = IMDP(address=address, noise_samples=noise_samples)

    all_labsets = {I.label[s] for s in I.states}
    # B = buchi_reach(all_labsets)
    B = Automata(all_labsets, "my_automaton.hoa")
    # B = Automata({tok for S in all_labsets for tok in S}, "my_automaton.hoa")
    print('will build product')
    P = Product(I, B)
    print('product is build')

    common_init_target = P.init_states & P.target
    common_init_losing = P.init_states & P.losing_sink
    common_target_losing = P.target & P.losing_sink

    print("Init ∩ Target:", len(common_init_target))
    print("Init ∩ Losing:", len(common_init_losing))
    print("Target ∩ Losing:", len(common_target_losing))

    only_init = P.init_states - (P.target | P.losing_sink)
    print("only init:", len(only_init))

    all_inits = len(P.init_states)
    print("all inits:", all_inits)

    res = quantitative_buchi_imdp(P, eps)
    res.update({"Qualitative_time_sec": P.qualitative_time_sec})
    update_csv_reslt(csv_path, address, res)
    return res


adds = [
'Ab_UAV_10-16-2025_20-48-14',
# 'Ab_UAV_10-16-2025_13-57-21',
# 'Ab_UAV_10-16-2025_15-11-36',
# 'Ab_UAV_10-16-2025_15-16-07',
# 'Ab_UAV_10-16-2025_15-25-59',
# 'Ab_UAV_10-16-2025_15-29-37'
]
# for add in adds:

add = adds[0]
res = run_imdp(address=add, noise_samples=20000, eps=1e-9)

mean_L_list = res["mean_L_list"]
mean_U_list = res["mean_U_list"]

x_values = list(range(0, len(mean_L_list) * iter_period, iter_period))

plt.plot(x_values, mean_L_list, marker='o', label='Mean Lower bound')
plt.plot(x_values, mean_U_list, marker='s', label='Mean Upper bound')

plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"Evolution_MeanL_MeanU_InitSt_VI_{add}.png")


