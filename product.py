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
        if not seeds:
            return set()
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
        if not self.target:
            return set(self.states)
        can_reach_target = self._backward_reachable(self.target)
        return set(self.states) - can_reach_target


