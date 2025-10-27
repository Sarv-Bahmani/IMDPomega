
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


class MDP:
    def __init__(self): #, filename):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.trans_MDP: Dict[Tuple[State, Action], Dict[State, float]] = {}
        self.label: Dict[State, Label] = {}


class IMDP:
    def __init__(self):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]] = {}
        self.label: Dict[State, Label] = {}


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


# class Product:
#     def __init__(self, mdp, buchi):
#         self.mdp = mdp
#         self.buchi = buchi
#         self.states: Set[ProdState] = set()
#         self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
#         self.trans_prod: Dict[Tuple[ProdState, Action], Dict[ProdState, float]] = {}
#         self.acc_states: Set[ProdState] = set()
#         self.graph = defaultdict(set)
#         self.build_product()
#         self.prod_graph()

#         self.mecs = self.mec_decomposition()
#         self.aecs = self.aecs_from_mecs(self.mecs)
#         self.target = set().union(*self.aecs) if self.aecs else set()
#         self.win_region = self.almost_sure_winning(self.target)


#     def build_product(self):
#         for s in self.mdp.states:
#             next_qs = self.buchi.step(self.buchi.q0, self.mdp.label[s]) | {self.buchi.q0}
#             for q_prime in next_qs:
#                 ps = (s, q_prime)
#                 self.states.add(ps)
#                 if q_prime in self.buchi.acc:
#                     self.acc_states.add(ps)
#         list_now = list(self.states)
#         for (s, q) in list_now:
#             self.trans_update(s, q)
#         list_after = list(self.states)
#         list_after = list(set(list_after) - set(list_now))
#         for (s, q) in list_after:
#             self.trans_update(s, q)

#     def trans_update(self, s, q):
#         for a in self.mdp.actions.get(s, ()):
#             outs = self.mdp.trans_MDP.get((s, a), {})
#             if not outs: continue
#             self.actions[(s, q)].add(a)
#             prod_outs = {}
#             for s2, prob in outs.items():
#                 for q3 in (self.buchi.step(q, self.mdp.label[s]) or {q}):
#                     ps = (s2, q3)
#                     self.states.add(ps)
#                     prod_outs[ps] = prod_outs.get(ps, 0.0) + prob
#                     if q3 in self.buchi.acc:
#                         self.acc_states.add(ps)
#             self.trans_prod[((s, q), a)] = prod_outs

#     def prod_graph(self) -> Dict[ProdState, Set[ProdState]]:
#         for (ps, a), outs in self.trans_prod.items():
#             for t, prob in outs.items():
#                 if prob > 0:
#                     self.graph[ps].add(t)

#     def sccs(self):
#         nodes = self.states
#         edges = self.graph
#         idx, low, st, on, comps = {}, {}, [], set(), []
#         i = 0
#         def dfs(v):
#             nonlocal i
#             idx[v] = i; low[v] = i; i += 1
#             st.append(v); on.add(v)
#             for w in edges.get(v, ()):
#                 if w not in idx:
#                     dfs(w); low[v] = min(low[v], low[w])
#                 elif w in on:
#                     low[v] = min(low[v], idx[w])
#             if low[v] == idx[v]:
#                 C = set()
#                 while True:
#                     w = st.pop(); on.remove(w); C.add(w)
#                     if w == v: break
#                 comps.append(C)
#         for v in nodes:
#             if v not in idx:
#                 dfs(v)
#         return comps

#     def closed_actions(self, SCC: Set[ProdState]) -> Dict[ProdState, Set[Action]]:
#         keep: Dict[ProdState, Set[Action]] = {}
#         for s in SCC:
#             kept = set()
#             for a in self.actions.get(s, ()):
#                 outs = self.trans_prod.get((s, a), {})
#                 if outs and all((t in SCC) for t in outs):
#                     kept.add(a)
#             if kept:
#                 keep[s] = kept
#         return keep

#     def mec_decomposition(self) -> List[Set[ProdState]]:
#         mecs: List[Set[ProdState]] = []
#         for C in self.sccs():
#             SCC = set(C)
#             changed = True
#             while changed:
#                 changed = False
#                 keep = self.closed_actions(SCC)
#                 drop = [s for s in SCC if s not in keep]
#                 if drop:
#                     for s in drop: SCC.remove(s)
#                     changed = True
#             if SCC:
#                 mecs.append(SCC)
#         mecs.sort(key=lambda X: -len(X))
#         maximal = []
#         for i, M1 in enumerate(mecs):
#             if any(M1 < M2 for j, M2 in enumerate(mecs) if j != i):
#                 continue
#             maximal.append(M1)
#         return maximal

#     def aecs_from_mecs(self, mecs: Iterable[Set[ProdState]]) -> List[Set[ProdState]]:
#         return [C for C in mecs if any(ps in self.acc_states for ps in C)]

#     def almost_sure_winning(self, Targ: Set[ProdState]) -> Set[ProdState]:
#         if not Targ:
#             return set()
#         prod_states = set(self.states)
#         changed = True
#         while changed:
#             changed = False
#             keep = self.closed_actions(prod_states)
#             to_remove = [state for state in prod_states if state not in keep]
#             if to_remove:
#                 for state in to_remove: prod_states.remove(state)
#                 changed = True
#         Region = set(Targ)
#         added = True
#         while added:
#             added = False
#             for statee in list(prod_states):
#                 if statee in Region:
#                     continue
#                 for a in self.actions.get(statee, ()):
#                     outs = self.trans_prod.get((statee, a), {})
#                     if outs:
#                         if all(t in prod_states for t in outs):
#                             if any(t in Region for t in outs):
#                                 Region.add(statee); added = True; break
#         return Region




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






if __name__ == "__main__":
    MDP = IMDP()
    # s0, s1, s2, s3 = 0, 1, 2, 3
    # MDP.states.update([s0, s1, s2, s3])

    # MDP.actions[s0].add("a")
    # MDP.actions[s1].update(["safe", "risky", "detour"])
    # MDP.actions[s2].update(["safe", "loop"])
    # MDP.actions[s3].add("a")

    # MDP.trans_MDP[(s0, "a")]      = {s0: 0.6, s1: 0.4}
    # MDP.trans_MDP[(s1, "safe")]   = {s0: 1.0}
    # MDP.trans_MDP[(s1, "risky")]  = {s3: 1.0}
    # MDP.trans_MDP[(s1, "detour")] = {s2: 1.0}
    # MDP.trans_MDP[(s2, "safe")]   = {s0: 1.0}
    # MDP.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    # MDP.trans_MDP[(s3, "a")]      = {s3: 1.0}

    # MDP.label[s0] = frozenset({"g"})
    # MDP.label[s1] = frozenset({"b"})
    # MDP.label[s2] = frozenset({"b"})
    # MDP.label[s3] = frozenset({"b"})


    # s0, s1, s2, s3 = 0, 1, 2, 3
    # MDP.states.update([s0, s1, s2, s3])

    # MDP.actions[s0].add("a")
    # MDP.actions[s0].update(["safe"])
    # MDP.actions[s1].update(["risky"])
    # MDP.actions[s2].update(["loop"])
    # MDP.actions[s3].add("safe")


    # MDP.trans_MDP[(s0, "a")]      = {s0: 0.1, s1: 0.9}
    # MDP.trans_MDP[(s1, "risky")]  = {s0: 0.000001, s2: 1-0.000001}
    # MDP.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    # MDP.trans_MDP[(s0, "safe")]   = {s3: 1.0}
    # MDP.trans_MDP[(s3, "safe")]   = {s3: 1.0}


    # MDP.label[s0] = frozenset({"g"})
    # MDP.label[s1] = frozenset({"b"})
    # MDP.label[s2] = frozenset({"b"})
    # MDP.label[s3] = frozenset({"g"})





    s0, s1, s2, s3 = 0, 1, 2, 3
    MDP.states.update([s0, s1, s2, s3])
    MDP.actions[s0].add("a"); MDP.actions[s1].update(["safe", "risky"]); MDP.actions[s2].add("a"); MDP.actions[s0].update(["safe"]); MDP.actions[s3].add("safe")
    MDP.label[s0] = frozenset({"g"}); MDP.label[s1] = frozenset({"b"}); MDP.label[s2] = frozenset({"b"}); MDP.label[s3] = frozenset({"g"})
    MDP.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    MDP.intervals[(s1, "risky")] = {s2: (0.9,0.9) , s0: (0.2,0.2)}
    MDP.intervals[(s2, "a")] = {s2: (1.0,1.0)}
    MDP.intervals[(s0, "safe")]   = {s3: (1.0,1.0)}
    MDP.intervals[(s3, "safe")]   = {s3: (1.0,1.0)}


    AP = {"g", "b"}
    Buchi = BuchiA(AP) # GF g
    Buchi.add_state(0, initial=True, accepting=False)   # q0
    Buchi.add_state(1, accepting=True)                  # q1
    all_labels = {MDP.label[st] for st in MDP.label}
    for lab in all_labels:
        if "g" in lab:
            Buchi.add_edge(0, lab, 1)
            Buchi.add_edge(1, lab, 1)
        else:
            Buchi.add_edge(0, lab, 0)
            Buchi.add_edge(1, lab, 0)



    Prod = Product(MDP, Buchi)
    res = {
        "product_states": Prod.states,
        "AECs": Prod.aecs,
        "target_union": Prod.target,
        "winning_product_states": Prod.win_region}


    print("#product states:", len(res["product_states"]))
    print("#AECs:", len(res["AECs"]))
    print("winning |W|:", len(res["winning_product_states"]))
    winners_base = {s for (s, q) in res["winning_product_states"]}
    print("base winners:", winners_base)

    print(Prod.target)

    print("Qualitative analysis time (sec):", Prod.qualitative_time_sec)
    print("Losing sink states:", Prod.losing_sink)

    # s0, s1, s2 = 0, 1, 2
    # I.states.update([s0, s1, s2])
    # I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a")
    # I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset({"b"}); I.label[s2] = frozenset({"b"})

    # I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    # I.intervals[(s1, "safe")]  = {s0: (1.0, 1.0)}
    # I.intervals[(s1, "risky")] = {s2: (0.6, 1.0) , s0: (0.0, 0.4)}
    # I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}

    # s0, s1, s2, s3 = 0, 1, 2, 3
    # I.states.update([s0, s1, s2, s3])
    # I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a"); I.actions[s0].update(["safe"]); I.actions[s3].add("safe")
    # I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset({"b"}); I.label[s2] = frozenset({"b"}); I.label[s3] = frozenset({"g"})

    # I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    # I.intervals[(s1, "safe")]  = {s0: (1.0, 1.0)}
    # I.intervals[(s1, "risky")] = {s2: (0.6, 1.0) , s0: (0.0, 0.4)}
    # I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}
    # I.intervals[(s0, "safe")]   = {s3: (1.0, 1.0)}
    # I.intervals[(s3, "safe")]   = {s3: (1.0, 1.0)}

    # w, m, f, t = 0, 1, 2, 3
    # s0, s1, s2, s3 = 0, 1, 2, 3
    # I.states.update([s0, s1, s2, s3])
    # I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a"); I.actions[s0].update(["safe"]); I.actions[s3].add("safe")
    # I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset({"b"}); I.label[s2] = frozenset({"b"}); I.label[s3] = frozenset({"g"})

    # I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    # I.intervals[(s1, "risky")] = {s2: (0.6, 0.9) , s0: (0.0, 0.4)}
    # I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}
    # I.intervals[(s0, "safe")]   = {s3: (0.6, 0.8) , s1: (0.0, 0.4)}
    # I.intervals[(s3, "safe")]   = {s3: (1.0, 1.0)}


