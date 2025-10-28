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



def load_sta_align(path_sta: str):
    ids: List[int] = []
    with open(path_sta) as f:
        _ = f.readline()                    # header like: (x_pos,x_vel,...)
        for line in f:
            m = re.match(r'(\d+):', line)
            if m: ids.append(int(m.group(1)))
    remap = {sid:i for i,sid in enumerate(sorted(ids))}
    return remap, len(remap)

def load_lab_align(path_lab: str, remap: Dict[int,int]):
    # first line: 0="init" 1="deadlock" 2="reached" 3="failed"
    with open(path_lab) as f:
        first = f.readline().strip()
        labmap: Dict[int,str] = {}
        for tok in first.split():
            m = re.match(r'(\d+)="([^"]+)"', tok)
            if m: labmap[int(m.group(1))] = m.group(2).lower()

        names_per_state: Dict[int, Set[str]] = defaultdict(set)
        for line in f:
            m = re.match(r'(\d+):\s*(.*)$', line.strip())
            if not m: continue
            s_orig = int(m.group(1))
            s = remap[s_orig]
            rest = m.group(2).strip()
            if not rest: continue
            for tok in rest.split():
                lid = int(tok)
                names_per_state[s].add(labmap.get(lid, ""))

    # Build FrozenSet labels for your IMDP
    label: Dict[int, FrozenSet[str]] = {}
    goal: Set[int] = set()
    avoid: Set[int] = set()
    init: Set[int]  = set()
    AtomicP: Set[int] = set()
    for s, ns in names_per_state.items():
        L = set(ns)
        label[s] = frozenset(L if L else {})
        AtomicP.update(lab for lab in L)
        if "reached" in L or "goal" in L or "target" in L:
            goal.add(s)
        if "failed" in L or "deadlock" in L or "unsafe" in L or "bad" in L:
            avoid.add(s)
        if "init" in L:
            init.add(s)
    return label, goal, avoid, init, AtomicP

def load_tra_align(path_tra: str, remap: Dict[int,int]):
    # header: "N  |SA|  |E|" (three integers) – we’ll just read and ignore
    with open(path_tra) as f:
        _ = f.readline()
        pat = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s*\[\s*([0-9.eE+\-]+)\s*,\s*([0-9.eE+\-]+)\s*\]')
        trans = defaultdict(dict)     # (s,a) -> {s':(l,u)}
        actions = defaultdict(set)    # s -> {a}
        for raw in f:
            line = raw.strip()
            if not line: continue
            m = pat.match(line)
            if not m:
                raise ValueError(f"Bad .tra line: {line[:120]}")
            s0, a, s1 = int(m.group(1)), str(int(m.group(2))), int(m.group(3))
            l, u = float(m.group(4)), float(m.group(5))
            s  = remap[s0]
            sp = remap[s1]
            actions[s].add(a)
            trans[(s, a)][sp] = (l, u)
    return actions, trans

def check_intervals(intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]]):
    for (s, a), outs in intervals.items():
        for l, u in outs.values():
            if u > 0 and l == 0:
                return False
    return True

def imdp_from_files_quant(sta_path: str, lab_path: str, tra_path: str, I) -> Dict[str, Set[int]]:
    # I is an instance of your IMDP() class from quant (1).py
    remap, n_states = load_sta_align(sta_path)
    # I.states = set(range(n_states))
    I.states.update([i for i in range(n_states)])
    I.label, goal, avoid, init, AtomicP = load_lab_align(lab_path, remap)
    I.actions, I.intervals = load_tra_align(tra_path, remap)
    is_SMDP = check_intervals(I.intervals)
    if not is_SMDP:
        raise ValueError("Some transition has upper > 0 but lower = 0")

    return {"reached": goal, "avoid": avoid, "init": init}, AtomicP

class IMDP:
    def __init__(self):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]] = {}
        self.label: Dict[State, Label] = {}




address='Ab_UAV_10-16-2025_20-48-14'
noise_samples=20000
base = root_models / address / f"N={noise_samples}_0"
sta_p = base / sta; lab_p = base / lab; tra_p = base / tra
I = IMDP()
print('WILL read the data')
_, _ = imdp_from_files_quant(str(sta_p), str(lab_p), str(tra_p), I)
print('data is read')
all_labsets_2_2 = {I.label[s] for s in I.states}














class BuchiA:
    def __init__(self, ap: Set[str]):
        self.ap = set(ap)
        self.Q: Set[QState] = set()
        self.init: Set[QState] = set()
        self.q0 = 0
        self.acc: Set[QState] = set()
        self.trans_automa: Dict[Tuple[QState, Label], Set[QState]] = defaultdict(set)

    def add_state(self, q, initial=False, accepting=False):
        self.Q.add(q)
        if initial: 
            self.q0 = q
            self.init.add(q)
        if accepting: 
            self.acc.add(q)

    def add_edge(self, q, lab, q2):
        self.trans_automa[(q, lab)].add(q2)

    def step(self, q, lab) -> Set[QState]:
        return self.trans_automa.get((q, lab), set())




def buchi_reach(all_labsets): 
    B = BuchiA({tok for S in all_labsets for tok in S}) 
    B.add_state(0, accepting=True) 
    B.add_state(1, initial=True) 
    B.add_state(2) 
    for labset in all_labsets: 
        B.add_edge(2, labset, 2) 
        if "failed" in labset: 
            B.add_edge(0, labset, 2) 
            B.add_edge(1, labset, 2) 
        elif "reached" in labset: # and not "deadlock" in labset and not "failed" in labset: 
            B.add_edge(0, labset, 0) 
            B.add_edge(1, labset, 0) 
        else: 
            B.add_edge(0, labset, 0) 
            B.add_edge(1, labset, 1) 
    return B





def parse_hoa_to_buchia(hoa_text: str) -> BuchiA:
    lines = hoa_text.strip().split('\n')
    ap_list = []
    start_states = set()
    
    i = 0
    while i < len(lines) and not lines[i].strip().startswith('--BODY--'):
        line = lines[i].strip()

        if line.startswith('States:'):
            num_states = int(line.split()[1])

        elif line.startswith('Start:'):
            start_states.add(int(line.split()[1]))        
        
        elif line.startswith('AP:'):
            parts = line.split()
            num_ap = int(parts[1])
            ap_list = []
            for j in range(2, 2 + num_ap):
                ap_name = parts[j].strip('"')
                ap_list.append(ap_name)
        
        i += 1
    
    B = BuchiA(set(ap_list))
    
    i += 1
    current_state = None
    while i < len(lines) and not lines[i].strip().startswith('--END--'):
        line = lines[i].strip()
        
        if line.startswith('State:'):
            parts = line.split()
            current_state = int(parts[1])
            
            is_accepting = (len(parts) > 2 and parts[2].startswith('{'))
            is_initial = (current_state in start_states)
            
            B.add_state(current_state, initial=is_initial, accepting=is_accepting)
            
        elif line.startswith('[') and current_state is not None:
            guard_end = line.index(']')
            guard = line[1:guard_end]
            dest_state = int(line[guard_end+1:].strip())
            labels = parse_guard_to_labels(guard, ap_list)
            for label in labels:
                B.add_edge(current_state, label, dest_state)
        
        i += 1
    
    return B


def parse_guard_to_labels(guard: str, ap_list: List[str]) -> Set[FrozenSet[str]]:
    """
    Guard uses AP indices: 0 = "failed", 1 = "reached"
    
    Examples:
    - "[!0&1]" means not failed AND reached
    - "[0]" means failed
    - "[t]" means true (any label)
    """
    if guard == 't':
        all_labels = set()
        for i in range(2 ** len(ap_list)):
            label = frozenset(ap_list[j] for j in range(len(ap_list)) if (i >> j) & 1)
            all_labels.add(label)
        return all_labels
    
    # Parse boolean expression
    # Replace AP indices with actual names for evaluation
    result_labels = set()
    
    # Generate all possible labelsets and test against guard
    for i in range(2 ** len(ap_list)):
        label = frozenset(ap_list[j] for j in range(len(ap_list)) if (i >> j) & 1)
        
        if evaluate_guard(guard, ap_list, label):
            result_labels.add(label)
    
    return result_labels


def evaluate_guard(guard: str, ap_list: List[str], label: FrozenSet[str]) -> bool:
    """
    Evaluate if a guard expression matches a given label.
    Guard format: uses indices (0, 1, ...) with !, &, | operators
    """
    # Create a mapping: "0" -> whether ap_list[0] is in label
    expr = guard
    
    # Replace AP indices with True/False based on label
    for idx, ap_name in enumerate(ap_list):
        is_present = ap_name in label
        # Replace index with boolean (be careful with order - replace higher indices first)
        expr = expr.replace(f'{idx}', str(is_present))
    
    expr = expr.replace('!', ' not ')
    expr = expr.replace('&', ' and ')
    expr = expr.replace('|', ' or ')
    
    try:
        return eval(expr)
    except:
        return False


# Usage:
hoa_text = """HOA: v1
name: "G!failed & Freached"
States: 3
Start: 1
AP: 2 "failed" "reached"
acc-name: Buchi
Acceptance: 1 Inf(0)
properties: trans-labels explicit-labels state-acc complete
properties: deterministic stutter-invariant very-weak
--BODY--
State: 0 {0}
[!0] 0
[0] 2
State: 1
[!0&1] 0
[!0&!1] 1
[0] 2
State: 2
[t] 2
--END--"""

B = parse_hoa_to_buchia(hoa_text)
print(f"States: {B.Q}")
print(f"Initial: {B.q0}")
print(f"Accepting: {B.acc}")
print(f"APs: {B.ap}")





B_PREVVV = buchi_reach(all_labsets_2_2)

a = 5