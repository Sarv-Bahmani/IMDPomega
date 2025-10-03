# imdp_buchi_quant.py
from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, FrozenSet, Iterable, List

# ---------- Types ----------
State = int
QState = int
Action = str
Label = FrozenSet[str]
ProdState = Tuple[State, QState]

# ---------- Base models ----------
class MDP:
    """Fixed-probabilities MDP. Use to build quick tests; can be converted to IMDP."""
    def __init__(self):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.P: Dict[Tuple[State, Action], Dict[State, float]] = {}  # only positive probs
        self.label: Dict[State, Label] = {}

class IMDP:
    """Interval MDP: transitions are intervals [l,u] with sum(l) <= 1 <= sum(u)."""
    def __init__(self):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]] = {}
        self.label: Dict[State, Label] = {}

    @staticmethod
    def from_mdp(M: MDP) -> "IMDP":
        I = IMDP()
        I.states = set(M.states)
        I.actions = defaultdict(set, {s: set(aset) for s, aset in M.actions.items()})
        I.label = dict(M.label)
        for key, succs in M.P.items():
            I.intervals[key] = {t: (p, p) for t, p in succs.items()}
        return I

# ---------- State-accepting Büchi automaton (deterministic over concrete labels) ----------
class BuchiSA:
    def __init__(self, ap: Iterable[str]):
        self.ap = set(ap)
        self.Q: Set[QState] = set()
        self.q0: QState | None = None
        self.F: Set[QState] = set()
        self.trans: Dict[Tuple[QState, Label], QState] = {}  # deterministic on exact labels

    def add_state(self, q: QState, *, initial: bool = False, accepting: bool = False):
        self.Q.add(q)
        if initial:
            self.q0 = q
        if accepting:
            self.F.add(q)

    def add_edge(self, q: QState, lab: Label, q2: QState):
        self.trans[(q, lab)] = q2

    def step(self, q: QState, lab: Label) -> QState | None:
        return self.trans.get((q, lab), None)

# ---------- Product (IMDP × BüchiSA) ----------
class ProductIMDP:
    def __init__(self):
        self.states: Set[ProdState] = set()
        self.actions: Dict[ProdState, Set[Action]] = defaultdict(set)
        # intervals[(x,a)][y] = (l,u) with y in product states
        self.intervals: Dict[Tuple[ProdState, Action], Dict[ProdState, Tuple[float, float]]] = {}
        self.acc_states: Set[ProdState] = set()  # accepting product states (q in F)

def build_product_state_accepting(I: IMDP, B: BuchiSA) -> ProductIMDP:
    assert B.q0 is not None, "Buchi automaton must have an initial state"
    P = ProductIMDP()

    # Seed product states: advance automaton from q0 on each model state's label
    for s in I.states:
        q2 = B.step(B.q0, I.label[s])
        if q2 is None:
            continue
        ps = (s, q2)
        P.states.add(ps)
        if q2 in B.F:
            P.acc_states.add(ps)

    # Lift transitions; BA reads the SUCCESSOR's label
    for (s, q2) in list(P.states):
        for a in I.actions.get(s, ()):
            succ_iv = I.intervals.get((s, a), {})
            if not succ_iv:
                continue
            P.actions[(s, q2)].add(a)
            out: Dict[ProdState, Tuple[float, float]] = {}
            for s2, (l, u) in succ_iv.items():
                q3 = B.step(q2, I.label[s2])
                if q3 is None:
                    continue
                t = (s2, q3)
                if t not in P.states:
                    P.states.add(t)
                    if q3 in B.F:
                        P.acc_states.add(t)
                # sum intervals if multiple concrete edges collapse to same (s2,q3)
                old = out.get(t, (0.0, 0.0))
                out[t] = (old[0] + l, old[1] + u)
            if out:
                P.intervals[((s, q2), a)] = out

    return P

# ---------- Graph utilities, MECs, AECs ----------
def underlying_edges(P: ProductIMDP) -> Dict[ProdState, Set[ProdState]]:
    E: Dict[ProdState, Set[ProdState]] = defaultdict(set)
    for (x, a), outs in P.intervals.items():
        for y, (l, u) in outs.items():
            if u > 0.0:
                E[x].add(y)
    return E

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
        if v not in idx: dfs(v)
    return comps

def closed_actions(P: ProductIMDP, S: Set[ProdState]) -> Dict[ProdState, Set[Action]]:
    keep: Dict[ProdState, Set[Action]] = {}
    for x in S:
        kept = set()
        for a in P.actions.get(x, ()):
            outs = P.intervals.get((x, a), {})
            # action is "closed" if all successors stay inside S and u>0
            if outs and all((y in S) for y in outs):
                kept.add(a)
        if kept:
            keep[x] = kept
    return keep

def mec_decomposition(P: ProductIMDP) -> List[Set[ProdState]]:
    E = underlying_edges(P)
    mecs: List[Set[ProdState]] = []
    for C in sccs(P.states, E):
        S = set(C)
        changed = True
        while changed:
            changed = False
            keep = closed_actions(P, S)
            drop = [x for x in list(S) if x not in keep]
            if drop:
                for x in drop: S.remove(x)
                changed = True
        if S:
            mecs.append(S)
    # keep only maximal by inclusion
    mecs.sort(key=lambda X: -len(X))
    maximal: List[Set[ProdState]] = []
    for i, M1 in enumerate(mecs):
        if any(M1 < M2 for j, M2 in enumerate(mecs) if j != i):  # proper subset
            continue
        maximal.append(M1)
    return maximal

def aecs_from_mecs(P: ProductIMDP, mecs: Iterable[Set[ProdState]]) -> List[Set[ProdState]]:
    return [C for C in mecs if any(x in P.acc_states for x in C)]

# ---------- Interval Expectations (Haddad–Monmege greedy min/max) ----------
def min_expectation_for_action(intervals_list: List[Tuple[ProdState, float, float]],
                               V: Dict[ProdState, float]) -> float:
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

def max_expectation_for_action(intervals_list: List[Tuple[ProdState, float, float]],
                               V: Dict[ProdState, float]) -> float:
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
def interval_iteration(P: ProductIMDP, T: Set[ProdState], eps: float = 1e-10, max_iter: int = 100000):
    # Initialize: targets are 1; others 0
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
                    iv = P.intervals.get((x, a), {})
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

# ---------- Büchi quantitative wrapper ----------
def quantitative_buchi_imdp(I: IMDP, B: BuchiSA, eps: float = 1e-10):
    P = build_product_state_accepting(I, B)
    mecs = mec_decomposition(P)
    aecs = aecs_from_mecs(P, mecs)
    T: Set[ProdState] = set().union(*aecs) if aecs else set()
    L, U = interval_iteration(P, T, eps=eps)
    return {
        "product": P,
        "AECs": aecs,
        "target_union": T,
        "L": L,   # minimal (robust) probabilities to satisfy Büchi
        "U": U,   # maximal (optimistic) probabilities to satisfy Büchi
    }

# ---------- Helpers to build common automata ----------
def buchi_GF_g_state_accepting() -> BuchiSA:
    """Two-state deterministic Büchi for GF g over AP={g,b} but we only check 'g'."""
    AP = {"g", "b"}
    B = BuchiSA(AP)
    B.add_state(0, initial=True, accepting=False)  # q0
    B.add_state(1, accepting=True)                 # q1
    labs = [frozenset(), frozenset({"g"}), frozenset({"b"}), frozenset({"g", "b"})]
    for lab in labs:
        if "g" in lab:
            B.add_edge(0, lab, 1)
            B.add_edge(1, lab, 1)
        else:
            B.add_edge(0, lab, 0)
            B.add_edge(1, lab, 0)
    return B

# ---------- Demos ----------
def demo_mdp():
    print("=== MDP demo (L == U) ===")
    # Two-state symmetric MDP; s0 labeled g, s1 unlabeled; action a flips/stays 0.5
    M = MDP()
    s0, s1 = 0, 1
    M.states.update([s0, s1])
    M.actions[s0].add("a"); M.actions[s1].add("a")
    M.P[(s0, "a")] = {s0: 0.5, s1: 0.5}
    M.P[(s1, "a")] = {s0: 0.5, s1: 0.5}
    M.label[s0] = frozenset({"g"}); M.label[s1] = frozenset()

    I = IMDP.from_mdp(M)
    B = buchi_GF_g_state_accepting()
    res = quantitative_buchi_imdp(I, B, eps=1e-12)
    P = res["product"]; L, U = res["L"], res["U"]
    # Project to base states: take supremum over automaton components
    proj_L = defaultdict(float); proj_U = defaultdict(float)
    for (s, q), v in L.items():
        proj_L[s] = max(proj_L[s], v)
    for (s, q), v in U.items():
        proj_U[s] = max(proj_U[s], v)
    print("L (min probs) by base state:", dict(proj_L))
    print("U (max probs) by base state:", dict(proj_U))

def demo_imdp():
    print("\n=== IMDP demo (L <= U, strict) ===")
    # States: s0 (g), s1 (neutral choice), s2 (trap, no g)
    # At s1, action 'risky' has uncertain prob to trap in [0.6,1.0] (to g in [0,0.4]),
    #        action 'safe' returns to s0 with [1,1].
    I = IMDP()
    s0, s1, s2 = 0, 1, 2
    I.states.update([s0, s1, s2])
    I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a")
    I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset(); I.label[s2] = frozenset()

    # s0: 0.5 stay, 0.5 go to s1
    I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    # s1: safe -> s0 w.p. 1; risky -> to trap with [0.6,1.0], to s0 with [0.0,0.4]
    I.intervals[(s1, "safe")]  = {s0: (1.0, 1.0)}
    I.intervals[(s1, "risky")] = {s2: (0.6, 1.0), s0: (0.0, 0.4)}
    # trap
    I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}

    B = buchi_GF_g_state_accepting()
    res = quantitative_buchi_imdp(I, B, eps=1e-12)
    L, U = res["L"], res["U"]

    # Project to base
    proj_L = defaultdict(float); proj_U = defaultdict(float)
    for (s, q), v in L.items(): proj_L[s] = max(proj_L[s], v)
    for (s, q), v in U.items(): proj_U[s] = max(proj_U[s], v)
    print("L (min probs) by base state:", dict(proj_L))  # robust (nature adversarial)
    print("U (max probs) by base state:", dict(proj_U))  # optimistic

if __name__ == "__main__":
    demo_mdp()
    demo_imdp()
