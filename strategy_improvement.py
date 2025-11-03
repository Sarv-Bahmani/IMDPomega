from imdp import IMDP
from automata import Automata
from product import Product
import random
from typing import Dict, Set, Tuple, FrozenSet

State = int
QState = int
Action = str
ProdState = Tuple[State, QState]
Label = FrozenSet[str]


def initialize_env_policy_random():
    env_policy: Dict[Tuple[ProdState, Action], Dict[ProdState, float]] = {}
    for (x, a), intervals in P.trans_prod.items():
        if x in P.losing_sink: 
            continue
        dist = {}
        if x in P.target:
            dist[x] = 1.0
            env_policy[(x, a)] = dist
            continue
        residual = 1.0
        for y, (l, u) in intervals.items():
            dist[y] = l
            residual -= l
        
        r = max(0.0, residual)
        while r > 0:
            chosen, (l, u) = random.choice(list(intervals.items()))
            add = min(u - dist[chosen], r)
            dist[chosen] += add
            r -= add

        env_policy[(x, a)] = dist

    return env_policy


def initialize_player_policy():
    player_policy: Dict[ProdState, Action] = {}
    for x in P.states:
        if x in P.target or x in P.losing_sink: continue
        actions = P.actions.get(x, [])
        if actions:
            player_policy[x] = random.choice(list(actions))

    return player_policy

def policy_evaluation(V, player_policy, env_policy, max_iter=10, eps=1e-1):
    for _ in range(max_iter):
        delta = 0.0
        for x in P.states:
            if x in P.target or x in P.losing_sink: continue
            
            a = player_policy.get(x)
            dist = env_policy.get((x, a), {})
            new_v = sum(prob * V.get(y, 0.0) for y, prob in dist.items())
            
            delta = max(delta, abs(new_v - V[x]))
            V[x] = new_v        
        if delta < eps:
            break
    return V

def player_policy_improvement(V, player_policy, env_policy):
    improved = False
    for x in P.states:
        if x in P.target or x in P.losing_sink:
            continue
        best_action = None
        best_value = -float('inf')
        for a in P.actions.get(x, []):
            dist = env_policy.get((x, a), {})
            exp_value = sum(prob * V.get(y, 0.0) for y, prob in dist.items())
            if exp_value > best_value:
                best_value = exp_value
                best_action = a
        if best_action and best_action != player_policy.get(x):
            player_policy[x] = best_action
            improved = True
    return improved


def env_policy_improvement(V, player_policy, env_policy):
    for (x, a), intervals in P.trans_prod.items():
        dist = {}
        residual = 1.0
        for y, (l, u) in intervals.items():
            dist[y] = l
            residual -= l
        r = max(0.0, residual)
        while r > 0:
            worst, (l, u) = min(intervals.items(), key=lambda item: V.get(item[0], 0.0))
            add = min(u - dist[worst], r)
            dist[worst] += add
            r -= add
        env_policy[(x, a)] = dist


def strategy_improvement(P, player_iters=5, max_outer_iters=100):
    env_policy = initialize_env_policy_random()
    player_policy = initialize_player_policy()
    V = {x: 0.0 for x in P.states}
    V.update({x: 1.0 for x in P.target})
    V.update({x: 0.0 for x in P.losing_sink})
    
    for outer in range(max_outer_iters):
        for _ in range(player_iters):
            V = policy_evaluation(V, player_policy, env_policy)
            improved = player_policy_improvement(V, player_policy, env_policy)
            if not improved:
                break
        
        V = policy_evaluation(V, player_policy, env_policy)
        env_policy_improvement(V, player_policy, env_policy)
    return V, player_policy, env_policy


address = 'Ab_UAV_10-16-2025_20-48-14'
noise_samples = 20000
I = IMDP(address=address, noise_samples=noise_samples)


all_labsets = {I.label[s] for s in I.states}
B = Automata(all_labsets, "my_automaton.hoa")
P = Product(I, B)

V, _, _ = strategy_improvement(P, player_iters=5, max_outer_iters=20)


# I = IMDP(address=address, noise_samples=noise_samples, read_from_files=False)
# s0, s1, s2, s3 = 0, 1, 2, 3
# I.states.update([s0, s1, s2, s3])
# I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a"); I.actions[s0].update(["safe"]); I.actions[s3].add("safe")
# I.label[s0] = frozenset({"init", "g"}) 
# I.label[s1] = frozenset({"failed"})
# I.label[s2] = frozenset({"failed"})
# I.label[s3] = frozenset({"g"})

# I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
# I.intervals[(s1, "risky")] = {s2: (0.6, 0.9) , s0: (0.0, 0.4)}
# I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}
# I.intervals[(s0, "safe")]   = {s3: (0.6, 0.8) , s1: (0.0, 0.4)}
# I.intervals[(s3, "safe")]   = {s3: (1.0, 1.0)}


