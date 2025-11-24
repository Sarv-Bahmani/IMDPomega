from collections import defaultdict
from typing import Dict, Set, Tuple, FrozenSet, List
import re
from pathlib import Path

import sys
sys.setrecursionlimit(200000)


sta = "Abstraction_interval.sta"
lab = "Abstraction_interval.lab"
tra = "Abstraction_interval.tra"

csv_path = Path("gen_imdp_info/IMDPs_info.csv")

State = int
QState = int
Action = str
ProdState = Tuple[State, QState]
Label = FrozenSet[str]


class IMDP:
    def __init__(self, model_type, address, noise_samples):
        self.states: Set[State] = set()
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]] = {}
        self.label: Dict[State, Label] = {}
        self.address = address
        self.noise_samples = noise_samples
        
        root_models = Path(f"data/raw/{model_type}")
        self.base = root_models / address / f"N={noise_samples}_0"
    
        self.imdp_from_files_quant()


    def load_sta_align(self, path_sta: str):
        ids: List[int] = []
        with open(path_sta) as f:
            _ = f.readline()                    # header like: (x_pos,x_vel,...)
            for line in f:
                m = re.match(r'(\d+):', line)
                if m: ids.append(int(m.group(1)))
        remap = {sid:i for i,sid in enumerate(sorted(ids))}
        return remap, len(remap)

    def load_lab_align(self, path_lab: str, remap: Dict[int,int]):
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

    def load_tra_align(self, path_tra: str, remap: Dict[int,int]):
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


    def check_intervals(self): #, intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]]):
        for (s, a), outs in self.intervals.items():
            for l, u in outs.values():
                if u > 0 and l == 0:
                    return False
        return True


    def imdp_from_files_quant(self): #, sta_path: str, lab_path: str, tra_path: str, I) -> Dict[str, Set[int]]:
        sta_path = self.base / sta; lab_path = self.base / lab; tra_path = self.base / tra
        # I is an instance of your IMDP() class from quant (1).py
        remap, n_states = self.load_sta_align(sta_path)
        # I.states = set(range(n_states))
        self.states.update([i for i in range(n_states)])
        self.label, goal, avoid, init, AtomicP = self.load_lab_align(lab_path, remap)
        self.actions, self.intervals = self.load_tra_align(tra_path, remap)
        is_SMDP = self.check_intervals()
        if not is_SMDP:
            raise ValueError("Some transition has upper > 0 but lower = 0")
        elif is_SMDP:
            print("This is a valid IMDP (no zero lower bounds with positive upper bounds).")
        return {"reached": goal, "avoid": avoid, "init": init}, AtomicP


