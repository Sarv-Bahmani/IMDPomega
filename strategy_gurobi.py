import gurobipy as gp
from imdp import IMDP
from automata import Automata
from product import Product
import random
from typing import Dict, Set, Tuple, FrozenSet
import time
import matplotlib.pyplot as plt


from datetime import datetime
def now_time():
    time = str(datetime.now().strftime("%H-%M"))
    return ("time: " + time)


address_str = "address"
val_iter_time_str = "Val_Iter_Execution_time_sec"
val_iter_converge_iter_str = "Val_Iter_Convergence_iteration"
qual_time_str = "Qualitative_time_sec"
transitions_str = 'Transitions'
Exported_States_PRISM_str = 'Exported States (PRISM)'
mean_V_list_str = "mean_V_list"

strat_imprv_Values_str = "Stratgy_Imprv_Values"
strat_imprv_Convergence_iteration_str = "Stratgy_Imprv_Convergence_iteration"
strat_imprv_Execution_time_sec_str = "Stratgy_Imprv_Execution_time_sec"


State = int
QState = int
Action = str
ProdState = Tuple[State, QState]
Label = FrozenSet[str]

iter_print = 1


def initialize_env_policy_random(P):
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

def solve_player_LP(env_policy, P):
    print("load model" + now_time())
    m = gp.Model("player_strategy")
    m.setParam('OutputFlag', 0)
    print("Gurobi model LOADED" + now_time())
    
    V = {}
    for x in P.states:
        if x in P.target:
            V[x] = m.addVar(lb=1.0, ub=1.0, name=f"V_{x}")
        elif x in P.losing_sink:
            V[x] = m.addVar(lb=0.0, ub=0.0, name=f"V_{x}")
        else:
            V[x] = m.addVar(lb=0.0, ub=1.0, name=f"V_{x}")
    
    print("will start defining expresions" + now_time())
    for x in P.states:
        if x in P.target or x in P.losing_sink:
            continue
            
        for action in P.actions.get(x, []):
            m.addConstr(
                V[x] >= sum(
                    prob * V[y] for y, prob in env_policy.get((x, action), {}).items()
                    )
                    ,name=f"constraint_{x}_{action}")

    print("defining expereseions DONE" + now_time())

    m.setObjective(gp.quicksum(V[s] for s in P.states 
                               if s not in P.target and s not in P.losing_sink), 
                   gp.GRB.MINIMIZE)
    print("will optimize..." + now_time())
    m.optimize()
    
    print("optimization DONE" + now_time())
    V_result = {s: v.X for s, v in V.items()}
    print("will extract optimal actions..." + now_time())
    player_strategy = extract_optimal_actions(V_result, env_policy, P)
    
    return V_result, player_strategy

def extract_optimal_actions(V_result, env_policy, P):
    player_strategy = {}
    for state in P.states:
        if state in P.target or state in P.losing_sink:
            continue
            
        best_action = None
        best_value = -float('inf')
        
        for action in P.actions.get(state, []):
            expected = 0.0
            for next_state, prob in env_policy.get((state, action), {}).items():
                expected += prob * V_result[next_state]
            
            if expected > best_value:
                best_value = expected
                best_action = action
        
        player_strategy[state] = best_action
    return player_strategy


def update_environment_policy(V, player_strategy, P):
    env_policy = {}
    
    for state in P.states:
        if state in P.target or state in P.losing_sink:
            continue
        
        actions = P.actions.get(state, ())
        # action = player_strategy[state] # **********************************
        for action in actions:
            intervals = P.trans_prod.get((state, action), {})
            
            dist = {}
            residual = 1.0
            
            for y, (l, u) in intervals.items():
                dist[y] = l
                residual -= l
            
            sorted_states = sorted(intervals.keys(), key=lambda s: V.get(s, 0))
            
            for y in sorted_states:
                l, u = intervals[y]
                add = min(u - dist[y], residual)
                dist[y] += add
                residual -= add
                if residual <= 0:
                    break
            
            env_policy[(state, action)] = dist
    
    return env_policy


def converged(V_old, V_new, tol):
    for state in V_old.keys():
        if abs(V_old[state] - V_new[state]) > tol:
            return False
    return True


def calc_init_mean(P, V):
    mean_i_V = []
    for (s, q) in P.init_states:
        mean_i_V.append(V[(s, q)])
    mean_V = sum(mean_i_V) / len(mean_i_V) if mean_i_V else 0.0
    return mean_V


def strategy_improve(P, eps):
    print("will start initializing env palicy" + now_time())
    env_policy = initialize_env_policy_random(P)
    print("initializing env palicy DONE" + now_time())
    V = {}
    for state in P.states:
        if   state in P.target     : V[state] = 1.0
        elif state in P.losing_sink: V[state] = 0.0
        else                       : V[state] = 0.5 #????????????????????????? 

    max_iterations = 51
    mean_V_list = []
    for iterator in range(max_iterations):
        # if iterator % iter_print == 0:
        print("Iteration:", iterator, now_time)
        mean_i_V = calc_init_mean(P, V)
        mean_V_list.append(mean_i_V)

        print("will go through solving LP..." + now_time())
        V_new, player_strategy = solve_player_LP(env_policy, P)
        
        print("will update env...." + now_time())
        env_policy = update_environment_policy(V_new, player_strategy, P)
        
        if converged(V, V_new, tol=eps):
            print(f"STRATEGY IMPROVEMENT breakkkkkk Converged at iteration: {iterator}, {now_time()}")
            V = V_new
            break
        V = V_new
    return V, iterator, mean_V_list


iter_init_save = 1
def plot_init_evolution_stra_impr(res, add):
    mean_V_list = res[mean_V_list_str]
    x_values = list(range(iter_init_save, (len(mean_V_list)+1) * iter_init_save, iter_init_save))
    plt.plot(x_values, mean_V_list, marker='o', label='Mean Initial States Value bound')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Evolution_InitStates_Strategy_Improvement_{add}.png")
    plt.close()


def strategy_improve_scope(P, eps):
    start_time = time.perf_counter()
    values, iterator, mean_V_list  = strategy_improve(P, eps=eps)
    execution_time = time.perf_counter() - start_time
    return {
        mean_V_list_str: mean_V_list,
        strat_imprv_Values_str: values,
        strat_imprv_Convergence_iteration_str: iterator,
        strat_imprv_Execution_time_sec_str: execution_time
    }


# results_strtgy = strategy_improve_scope(P, eps=1e-9)