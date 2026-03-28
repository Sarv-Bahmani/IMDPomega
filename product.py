import collections
from collections import deque, defaultdict
import time
from typing import Dict, Set, Tuple, FrozenSet, Iterable, List
from pathlib import Path


import sys
sys.setrecursionlimit(200000)


root_models = Path("MDPs")

State = int
QState = int
Action = str
ProdAction = Tuple[Action, QState]
ProdState = Tuple[State, QState]
Label = FrozenSet[str]


class Product:
    def __init__(self, imdp, buchi):
        self.imdp = imdp
        self.buchi = buchi
        self.states: Set[ProdState] = set()
        self.init_states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[ProdAction]] = defaultdict(set)
        self.trans_prod: Dict[Tuple[ProdState, ProdAction], Dict[ProdState, Tuple[float, float]]] = {}
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
        # reset, in case you ever rebuild
        self.states = set()
        self.init_states = set()
        self.actions = defaultdict(set)
        self.trans_prod = {}
        self.acc_states = set()

        # 1) build initial product states only
        for s in self.imdp.states:
            if "init" not in self.imdp.label.get(s, frozenset()):
                continue
            for q in self.buchi.init:
                ps = (s, q)
                self.init_states.add(ps)
                self.states.add(ps)
                if q in self.buchi.acc:
                    self.acc_states.add(ps)

        # 2) BFS from initial product states
        queue = deque(self.init_states)
        visited = set(self.init_states)

        while queue:
            (s, q) = queue.popleft()

            # build outgoing transitions of this reachable state
            succs = self.trans_update(s, q)

            # any newly discovered successor gets queued
            for ps2 in succs:
                if ps2 not in visited:
                    visited.add(ps2)
                    self.states.add(ps2)
                    if ps2[1] in self.buchi.acc:
                        self.acc_states.add(ps2)
                    queue.append(ps2)



    def trans_update(self, s, q):
        discovered_succs = set()
        for a in self.imdp.actions.get(s, ()):
            outs = self.imdp.intervals.get((s, a), {})
            if not outs:    continue
            labset = self.imdp.label.get(s, frozenset()) & self.buchi.ap
            next_qs = self.buchi.step(q, labset)
            for q3 in next_qs:
                prod_outs = {}
                self.actions[(s, q)].add((a,q3))
                for s2, (l, u) in outs.items():
                    ps = (s2, q3)
                    discovered_succs.add(ps)
                    old = prod_outs.get(ps, (0.0, 0.0))
                    prod_outs[ps] = (old[0] + l, old[1] + u)
                self.trans_prod[((s, q), (a,q3))] = prod_outs
        return discovered_succs

    # def trans_update(self, s, q):
    #     discovered_succs = set()
    #     for a in self.imdp.actions.get(s, ()):
    #         outs = self.imdp.intervals.get((s, a), {})
            
    #         if not outs:    continue
    #         for s2, (l, u) in outs.items():
    #             labset = self.imdp.label.get(s2, frozenset()) & self.buchi.ap
    #             next_qs = self.buchi.step(q, labset)
    #             for q3 in next_qs:
    #                 prod_outs = {}
    #                 self.actions[(s, q)].add((a,q3))
    #                 ps = (s2, q3)
    #                 discovered_succs.add(ps)
    #                 old = prod_outs.get(ps, (0.0, 0.0))
    #                 prod_outs[ps] = (old[0] + l, old[1] + u)
    #                 self.trans_prod[((s, q), (a,q3))] = prod_outs
        # return discovered_succs



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

    def closed_actions(self, SCC: Set[ProdState]) -> Dict[ProdState, Set[ProdAction]]:
        keep: Dict[ProdState, Set[ProdAction]] = {}
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


