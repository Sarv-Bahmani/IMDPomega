import json
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, FrozenSet, Iterable, List

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

        # self.mdp_js = self.load_mdp_from_file(filename)
        # self.mdp_parser()

    # def load_mdp_from_file(self, filename=None):
    #     file_addr = filename
    #     with open(file_addr, 'r') as f:
    #         return json.load(f)
    #
    # def mdp_parser(self):
    #     self.states = self.mdp_js[states_str]
    #     self.init_state = self.mdp_js[init_state_str]
    #     self.actions = self.mdp_js[actions_str]
    #     self.trans_MDP = self.mdp_js[trans_MDP_str]

class BuchiA:
    def __init__(self, ap: Set[str]):
        self.ap = set(ap)
        self.Q: Set[QState] = set()
        self.q0: QState = 0
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


class Product:
    def __init__(self, mdp, buchi):
        self.mdp = mdp
        self.buchi = buchi
        self.states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
        self.trans_prod: Dict[Tuple[ProdState, Action], Dict[ProdState, float]] = {}
        self.acc_states: Set[ProdState] = set()
        self.graph = defaultdict(set)
        self.build_product()
        self.prod_graph()

        self.mecs = self.mec_decomposition()
        self.aecs = self.aecs_from_mecs(self.mecs)
        self.target = set().union(*self.aecs) if self.aecs else set()
        self.win_region = self.almost_sure_winning(self.target)




    def build_product(self):
        for s in self.mdp.states:
            next_qs = self.buchi.step(self.buchi.q0, self.mdp.label[s]) | {self.buchi.q0}
            for q_prime in next_qs:
                ps = (s, q_prime)
                self.states.add(ps)
                if q_prime in self.buchi.acc:
                    self.acc_states.add(ps)
        list_now = list(self.states)
        for (s, q) in list_now:
            self.trans_update(s, q)
        list_after = list(self.states)
        list_after = list(set(list_after) - set(list_now))
        for (s, q) in list_after:
            self.trans_update(s, q)

    def trans_update(self, s, q):
        for a in self.mdp.actions.get(s, ()):
            outs = self.mdp.trans_MDP.get((s, a), {})
            if not outs: continue
            self.actions[(s, q)].add(a)
            prod_outs = {}
            for s2, prob in outs.items():
                for q3 in (self.buchi.step(q, self.mdp.label[s]) or {q}):
                    ps = (s2, q3)
                    self.states.add(ps)
                    prod_outs[ps] = prod_outs.get(ps, 0.0) + prob
                    if q3 in self.buchi.acc:
                        self.acc_states.add(ps)
            self.trans_prod[((s, q), a)] = prod_outs

    def prod_graph(self) -> Dict[ProdState, Set[ProdState]]:
        for (ps, a), outs in self.trans_prod.items():
            for t, prob in outs.items():
                if prob > 0:
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

