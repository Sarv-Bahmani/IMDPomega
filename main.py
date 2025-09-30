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

    def add_state(self, q: QState, initial=False, accepting=False):
        self.Q.add(q)
        if initial: self.q0 = q
        if accepting: self.acc.add(q)

    def add_edge(self, q: QState, lab: Label, q2: QState):
        self.trans_automa[(q, lab)].add(q2)

    def step(self, q: QState, lab: Label) -> Set[QState]:
        return self.trans_automa.get((q, lab), set())


class Product:
    def __init__(self, mdp: MDP, buchi: BuchiA):
        self.mdp = mdp
        self.buchi = buchi
        self.states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
        self.trans_prod: Dict[Tuple[ProdState, Action], Dict[ProdState, float]] = {}
        self.acc_states: Set[ProdState] = set()
        self.graph = defaultdict(set)
        self.build_product()
        self.prod_graph()


    def build_product(self):
        for s in self.mdp.states:
            next_qs = self.buchi.step(self.buchi.q0, self.mdp.label[s]) or {self.buchi.q0}
            for q_prime in next_qs:
                ps = (s, q_prime)
                self.states.add(ps)
                if q_prime in self.buchi.acc:
                    self.acc_states.add(ps)

        for (s, q) in list(self.states):
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



def sccs(nodes: Set[ProdState], edges: Dict[ProdState, Set[ProdState]]) -> List[Set[ProdState]]:
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

def closed_actions(P: Product, S: Set[ProdState]) -> Dict[ProdState, Set[Action]]:
    keep: Dict[ProdState, Set[Action]] = {}
    for s in S:
        kept = set()
        for a in P.actions.get(s, ()):
            outs = P.trans_prod.get((s, a), {})
            if outs and all((t in S) for t in outs):
                kept.add(a)
        if kept:
            keep[s] = kept
    return keep

# ---------- MECs and AECs ----------
def mec_decomposition(P: Product) -> List[Set[ProdState]]:
    E = P.graph
    mecs: List[Set[ProdState]] = []
    for C in sccs(P.states, E):
        S = set(C)
        changed = True
        while changed:
            changed = False
            keep = closed_actions(P, S)
            drop = [s for s in S if s not in keep]
            if drop:
                for s in drop: S.remove(s)
                changed = True
        if S:
            mecs.append(S)
    # Keep only maximal by inclusion
    mecs.sort(key=lambda X: -len(X))
    maximal = []
    for i, M1 in enumerate(mecs):
        if any(M1 < M2 for j, M2 in enumerate(mecs) if j != i):
            continue
        maximal.append(M1)
    return maximal

def aecs_from_mecs(P: Product, mecs: Iterable[Set[ProdState]]) -> List[Set[ProdState]]:
    return [C for C in mecs if any(ps in P.acc_states for ps in C)]

# ---------- Qualitative almost-sure reachability to T ----------
def almost_sure_winning(P: Product, T: Set[ProdState]) -> Set[ProdState]:
    if not T:
        return set()
    # 1) Greatest "safe" set S: there exists an action whose support stays inside S
    S = set(P.states)
    changed = True
    while changed:
        changed = False
        keep = closed_actions(P, S)
        to_remove = [s for s in S if s not in keep]
        if to_remove:
            for s in to_remove: S.remove(s)
            changed = True
    # 2) Restrict to states that can (controller-wise) reach T without leaving S
    R = set(T)
    added = True
    while added:
        added = False
        for s in list(S):
            if s in R:
                continue
            for a in P.actions.get(s, ()):
                outs = P.trans_prod.get((s, a), {})
                if outs and all(t in S for t in outs) and any(t in R for t in outs):
                    R.add(s); added = True; break
    return R

# ---------- Full pipeline for qualitative Büchi on MDP ----------
def qualitative_buchi_mdp(M: MDP, B: BuchiA):
    P = Product(M, B)
    mecs = mec_decomposition(P)
    aecs = aecs_from_mecs(P, mecs)
    T = set().union(*aecs) if aecs else set()
    W = almost_sure_winning(P, T)
    return {
        "product_states": P.states,
        "AECs": aecs,
        "target_union": T,
        "winning_product_states": W,
    }



def add_total_loops_for_labels(B: BuchiA, labels: Iterable[Label]):
    for lab in labels:
        B.add_edge(0, lab, 0)

# ---------- Demo ----------
if __name__ == "__main__":
    # Your 2-state model (s0 labelled g, s1 labelled b), each step flips/stays with prob 1/2.
    M = MDP()
    s0, s1 = 0, 1
    M.states.update([s0, s1])
    M.actions[s0].add("a"); M.actions[s1].add("a")
    M.trans_MDP[(s0, "a")] = {s0: 0.5, s1: 0.5}
    M.trans_MDP[(s1, "a")] = {s0: 0.5, s1: 0.5}
    M.label[s0] = frozenset({"g"})
    M.label[s1] = frozenset({"b"})

    AP = {"g", "b"}

    B = BuchiA(AP)
    B.add_state(0, initial=True)


    # Add the exact labels we will encounter in M:
    add_total_loops_for_labels(B, {M.label[s0], M.label[s1]})

    res = qualitative_buchi_mdp(M, B)
    print("#product states:", len(res["product_states"]))
    print("#AECs:", len(res["AECs"]))
    print("winning |W|:", len(res["winning_product_states"]))
    # Show base-state projection of winners:
    winners_base = {s for (s, q) in res["winning_product_states"]}
    print("base winners:", winners_base)
