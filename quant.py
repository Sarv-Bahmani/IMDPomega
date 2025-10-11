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

class IMDP:
    def __init__(self): #, filename):
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


class Product:
    def __init__(self, imdp, buchi):
        self.imdp = imdp
        self.buchi = buchi
        self.states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
        self.trans_prod: Dict[Tuple[ProdState, Action], Dict[ProdState, Tuple[float, float]]] = {}
        self.acc_states: Set[ProdState] = set()
        self.graph = defaultdict(set)
        self.build_product()
        self.prod_graph()

        self.mecs = self.mec_decomposition()
        self.aecs = self.aecs_from_mecs(self.mecs)
        self.target = set().union(*self.aecs) if self.aecs else set()
        self.win_region = self.almost_sure_winning(self.target)


    def build_product(self):
        for s in self.imdp.states:
            next_qs = self.buchi.step(self.buchi.q0, self.imdp.label[s]) | {self.buchi.q0}
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
        for a in self.imdp.actions.get(s, ()):
            outs = self.imdp.intervals.get((s, a), {})
            if not outs: continue
            self.actions[(s, q)].add(a)
            prod_outs: Dict[ProdState, Tuple[float, float]] = {}
            for s2, (l, u) in outs.items():
                for q3 in (self.buchi.step(q, self.imdp.label[s]) or {q}):
                    ps = (s2, q3)
                    self.states.add(ps)

                    # prod_outs[ps] = prod_outs.get(ps, 0.0) + prob
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





# ---------- Interval Expectations (Haddad–Monmege greedy min/max) ----------
def min_expectation_for_action(intervals_list: List[Tuple[ProdState, float, float]], V: Dict[ProdState, float]) -> float:
    # Start at lower bounds
    base = 0.0
    residual = 1.0
    items: List[Tuple[ProdState, float, float, float]] = []  # (y,l,u,V[y])
    for y, l, u in intervals_list:
        base += l * V.get(y, 0.0)
        residual -= l
        items.append((y, l, u, V.get(y, 0.0)))
    # Greedy: push residual to LOWEST V first
    items.sort(key=lambda t: t[3])  # ascending V
    exp = base
    r = max(0.0, residual)
    for y, l, u, vy in items:
        if r <= 0: break
        add = min(u - l, r)
        exp += add * vy
        r -= add
    return exp

def max_expectation_for_action(intervals_list: List[Tuple[ProdState, float, float]], V: Dict[ProdState, float]) -> float:
    base = 0.0
    residual = 1.0
    items: List[Tuple[ProdState, float, float, float]] = []
    for y, l, u in intervals_list:
        base += l * V.get(y, 0.0)
        residual -= l
        items.append((y, l, u, V.get(y, 0.0)))
    # Greedy: push residual to HIGHEST V first
    items.sort(key=lambda t: -t[3])  # descending V
    exp = base
    r = max(0.0, residual)
    for y, l, u, vy in items:
        if r <= 0: break
        add = min(u - l, r)
        exp += add * vy
        r -= add
    return exp

# ---------- Interval Iteration (returns L, U) ----------
def interval_iteration(P, T: Set[ProdState], eps = 1e-3, max_iter = 100):
    L: Dict[ProdState, float] = {x: (1.0 if x in T else 0.0) for x in P.states}
    U: Dict[ProdState, float] = {x: (1.0 if x in T else 0.0) for x in P.states}

    for _ in range(max_iter):
        deltaL = 0.0
        deltaU = 0.0

        for x in P.states:
            if x in T:
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
                    mexp = min_expectation_for_action(iv_list, L)
                    Mexp = max_expectation_for_action(iv_list, U)
                    best_min = mexp if best_min is None else max(best_min, mexp)
                    best_max = Mexp if best_max is None else max(best_max, Mexp)
                newL = best_min if best_min is not None else 0.0
                newU = best_max if best_max is not None else 0.0

            deltaL = max(deltaL, abs(newL - L[x]))
            deltaU = max(deltaU, abs(newU - U[x]))
            L[x], U[x] = newL, newU

        gap = max(U[x] - L[x] for x in P.states) if P.states else 0.0
        if max(deltaL, deltaU) <= eps and gap <= eps:
            break

    return L, U

def quantitative_buchi_imdp(P, eps: float = 1e-10):
    L, U = interval_iteration(P, P.target, eps=eps)
    return {
        "product": P,
        "AECs": P.aecs,
        "target_union": P.target,
        "L": L,   # minimal (robust) probabilities to satisfy Büchi
        "U": U,   # maximal (optimistic) probabilities to satisfy Büchi
    }



if __name__ == "__main__":

    print("\n=== IMDP demo (L <= U, strict) ===")
    I = IMDP()
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
    s0, s1, s2, s3 = 0, 1, 2, 3
    I.states.update([s0, s1, s2, s3])
    I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a"); I.actions[s0].update(["safe"]); I.actions[s3].add("safe")
    I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset({"b"}); I.label[s2] = frozenset({"b"}); I.label[s3] = frozenset({"g"})

    I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    I.intervals[(s1, "risky")] = {s2: (0.6, 0.9) , s0: (0.0, 0.4)}
    I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}
    I.intervals[(s0, "safe")]   = {s3: (1.0, 1.0)}
    I.intervals[(s3, "safe")]   = {s3: (1.0, 1.0)}




    AP = {"g", "b"}
    B = BuchiA(AP)
    B.add_state(0, initial=True, accepting=False)  # q0
    B.add_state(1, accepting=True)  # q1
    labs = [frozenset({lab}) for lab in AP]
    for lab in labs:
        if "g" in lab:
            B.add_edge(0, lab, 1)
            B.add_edge(1, lab, 1)
        else:
            B.add_edge(0, lab, 0)
            B.add_edge(1, lab, 0)

    P = Product(I, B)
    res = quantitative_buchi_imdp(P, eps=1e-12)
    L, U = res["L"], res["U"]


    proj_L = defaultdict(float); proj_U = defaultdict(float)
    for (s, q), v in L.items(): proj_L[s] = max(proj_L[s], v)
    for (s, q), v in U.items(): proj_U[s] = max(proj_U[s], v)
    print("L (min probs) by base state:", dict(proj_L))
    print("U (max probs) by base state:", dict(proj_U))
