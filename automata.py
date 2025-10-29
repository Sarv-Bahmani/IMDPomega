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


hoa_path = Path("hoa_files")



class Automata:
    def __init__(self, ap: Set[str], hoa_name):
        self.ap = set(ap)
        # self.ap = {tok for S in ap for tok in S}
        self.Q: Set[QState] = set()
        self.init: Set[QState] = set()
        self.q0 = 0
        self.acc: Set[QState] = set()
        self.trans_automa: Dict[Tuple[QState, Label], Set[QState]] = defaultdict(set)
        hoa_address = hoa_path / hoa_name
        with open(hoa_address) as f:
            hoa_text = f.read()
        # self.parse_hoa_to_auto(hoa_text)
        self.buchi_reach()

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



    def parse_hoa_to_auto(self, hoa_text: str):
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
        self.ap = set(ap_list)
        
        i += 1
        current_state = None
        while i < len(lines) and not lines[i].strip().startswith('--END--'):
            line = lines[i].strip()
            
            if line.startswith('State:'):
                parts = line.split()
                current_state = int(parts[1])
                
                is_accepting = (len(parts) > 2 and parts[2].startswith('{'))
                is_initial = (current_state in start_states)
                
                self.add_state(current_state, initial=is_initial, accepting=is_accepting)
                
            elif line.startswith('[') and current_state is not None:
                guard_end = line.index(']')
                guard = line[1:guard_end]
                dest_state = int(line[guard_end+1:].strip())
                labels = self.parse_guard_to_labels(guard, ap_list)
                for label in labels:
                    self.add_edge(current_state, label, dest_state)
            
            i += 1
        # return B


    def parse_guard_to_labels(self, guard: str, ap_list: List[str]) -> Set[FrozenSet[str]]:
        if guard == 't':
            all_labels = set()
            for i in range(2 ** len(ap_list)):
                label = frozenset(ap_list[j] for j in range(len(ap_list)) if (i >> j) & 1)
                all_labels.add(label)
            return all_labels
        result_labels = set()
        for i in range(2 ** len(ap_list)):
            label = frozenset(ap_list[j] for j in range(len(ap_list)) if (i >> j) & 1)
            
            if self.evaluate_guard(guard, ap_list, label):
                result_labels.add(label)
        
        return result_labels


    def evaluate_guard(self, guard: str, ap_list: List[str], label: FrozenSet[str]) -> bool:
        expr = guard
        for idx, ap_name in enumerate(ap_list):
            is_present = ap_name in label
            expr = expr.replace(f'{idx}', str(is_present))
        
        expr = expr.replace('!', ' not ')
        expr = expr.replace('&', ' and ')
        expr = expr.replace('|', ' or ')
        
        try:
            return eval(expr)
        except:
            return False







    def buchi_reach(self): 
        self.add_state(0) 
        self.add_state(1, initial=True, accepting=True) 
        self.add_state(2) 
        for labset in self.ap: 
            self.add_edge(2, labset, 2) 
            if "failed" in labset: 
                self.add_edge(0, labset, 2) 
                self.add_edge(1, labset, 2) 
            elif "reached" in labset: # and not "deadlock" in labset and not "failed" in labset: 
                self.add_edge(0, labset, 1) 
                self.add_edge(1, labset, 1) 
            else: 
                self.add_edge(0, labset, 0) 
                self.add_edge(1, labset, 1) 

