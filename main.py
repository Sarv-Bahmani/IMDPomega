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
            next_qs = self.buchi.step(self.buchi.q0, self.mdp.label[s]) | {self.buchi.q0}
            for q_prime in next_qs:
                ps = (s, q_prime)
                self.states.add(ps)
                if q_prime in self.buchi.acc:
                    self.acc_states.add(ps)

        listnow = list(self.states)
        for (s, q) in listnow:
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







# ---------- MECs and AECs ----------
def mec_decomposition(P: Product) -> List[Set[ProdState]]:
    mecs: List[Set[ProdState]] = []
    for C in P.sccs():
        SCC = set(C)
        changed = True
        while changed:
            changed = False
            keep = P.closed_actions(SCC)
            drop = [s for s in SCC if s not in keep]
            if drop:
                for s in drop: SCC.remove(s)
                changed = True
        if SCC:
            mecs.append(SCC)
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
def qualitative_buchi_mdp(MDP, BuchiA):
    Prod = Product(MDP, BuchiA)
    mecs = mec_decomposition(Prod)
    aecs = aecs_from_mecs(Prod, mecs)
    T = set().union(*aecs) if aecs else set()
    W = almost_sure_winning(Prod, T)
    return {
        "product_states": Prod.states,
        "AECs": aecs,
        "target_union": T,
        "winning_product_states": W,
    }



def add_total_loops_for_labels(B: BuchiA, labels: Iterable[Label]):
    for lab in labels:
        B.add_edge(0, lab, 0)





if __name__ == "__main__":
    mdp = MDP()
    s0, s1, s2, s3 = 0, 1, 2, 3
    mdp.states.update([s0, s1, s2, s3])

    mdp.actions[s0].add("a")
    mdp.actions[s1].update(["safe", "risky", "detour"])
    mdp.actions[s2].update(["safe", "loop"])
    mdp.actions[s3].add("a")

    mdp.trans_MDP[(s0, "a")]      = {s0: 0.6, s1: 0.4}
    mdp.trans_MDP[(s1, "safe")]   = {s0: 1.0}
    mdp.trans_MDP[(s1, "risky")]  = {s3: 1.0}
    mdp.trans_MDP[(s1, "detour")] = {s2: 1.0}
    mdp.trans_MDP[(s2, "safe")]   = {s0: 1.0}
    mdp.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    mdp.trans_MDP[(s3, "a")]      = {s3: 1.0}

    mdp.label[s0] = frozenset({"g"})
    mdp.label[s1] = frozenset({"b"})
    mdp.label[s2] = frozenset({"b"})
    mdp.label[s3] = frozenset({"b"})

    AP = {"g", "b"}
    buchi = BuchiA(AP) # GF g
    buchi.add_state(0, initial=True, accepting=False)   # q0
    buchi.add_state(1, accepting=True)                  # q1
    all_labels = {mdp.label[s0], mdp.label[s1], mdp.label[s2], mdp.label[s3]}
    for lab in all_labels:
        if "g" in lab:
            buchi.add_edge(0, lab, 1)
            buchi.add_edge(1, lab, 1)
        else:
            buchi.add_edge(0, lab, 0)
            buchi.add_edge(1, lab, 0)


    # ---------- Run your pipeline ----------
    res = qualitative_buchi_mdp(mdp, buchi)

    print("#product states:", len(res["product_states"]))
    print("#AECs:", len(res["AECs"]))
    print("winning |W|:", len(res["winning_product_states"]))
    winners_base = {s for (s, q) in res["winning_product_states"]}
    print("base winners (expect {0,1,2}):", winners_base)
