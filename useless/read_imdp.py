import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class IMDP:
    def __init__(self, n_states:int):
        self.states = set()
        self.states.update([i for i in range(n_states)])

        self.actions: Dict[int, Set[int]] = defaultdict(set)  # Act(s)
        # trans[(s,a)] = list of (s', l, u)
        self.intervals: Dict[Tuple[int, int], Dict[int, Tuple[float, float]]] = {}
        # self.goal: Set[int] = set()
        # self.avoid: Set[int] = set()
        # self.init: Set[int] = set()

# ---------- .sta ----------
def load_sta(path_sta: str) -> Tuple[List[int], Dict[int,int]]:
    """
    Your .sta format:
      (x_pos,x_vel,y_pos,y_vel,z_pos,z_vel)
      0:(-3,-3,-3,-3,-3,-3)
      1:(-2,-2,-2,-2,-2,-2)
      ...
    Returns:
      ids:   list of normalized ids 0..N-1 in ascending original-id order
      remap: map original-id -> normalized-id
    """
    ids: List[int] = []
    with open(path_sta) as f:
        header = f.readline()  # discard "(x_pos, ...)"
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'(\d+):', line)
            if not m:
                continue
            ids.append(int(m.group(1)))
    # Build 0..N-1 normalization (identity here, but robust if original ids have gaps)
    remap: Dict[int,int] = {sid: i for i, sid in enumerate(sorted(ids))}
    # Return normalized ids in ascending original-id order
    return [remap[sid] for sid in sorted(ids)], remap

# ---------- .lab ----------
def load_lab(path_lab: str, remap: Dict[int,int]) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    First line is a label dictionary like:
      0="init" 1="deadlock" 2="reached" 3="failed"
    Then lines:
      stateId: <space-separated label-ids>
    """
    goal, avoid, init = set(), set(), set()

    with open(path_lab) as f:
        first = f.readline().strip()
        # parse dictionary
        labmap: Dict[int,str] = {}
        for tok in first.split():
            m = re.match(r'(\d+)="([^"]+)"', tok)
            if m:
                labmap[int(m.group(1))] = m.group(2).lower()

        for line in f:
            line = line.strip()
            if not line or line.startswith("..."):  # your file has "..."
                continue
            m = re.match(r'(\d+):\s*(.*)$', line)
            if not m:
                continue
            orig = int(m.group(1))
            s = remap[orig]
            rest = m.group(2).strip()
            if not rest:
                continue
            lab_ids = [int(x) for x in rest.split()]
            names = {labmap.get(i, "") for i in lab_ids}

            # Map names to sets (exact to your file, but keep generic fallbacks)
            if {"reached"} & names or {"goal", "target"} & names:
                goal.add(s)
            if {"failed", "deadlock"} & names or {"unsafe", "bad"} & names:
                avoid.add(s)
            if "init" in names:
                init.add(s)

    return goal, avoid, init

# ---------- .tra ----------
def load_tra_intervals(path_tra: str, remap: Dict[int,int]) -> Tuple[Dict[Tuple[int,int], List[Tuple[int,float,float]]], Dict[int, Set[int]]]:
    """
    First line: three integers (metadata) e.g. "787 11260 525081"
    Then lines: s a s' [l,u]
    """
    # trans: Dict[Tuple[int,int], List[Tuple[int,float,float]]] = defaultdict(list)
    trans: Dict[Tuple[int, int], Dict[int, Tuple[float, float]]] = {}
    actions: Dict[int, Set[int]] = defaultdict(set)

    num_header = (0, 0, 0)
    with open(path_tra) as f:
        header = f.readline().strip()
        hdr_parts = header.split()
        if len(hdr_parts) >= 3 and all(p.isdigit() for p in hdr_parts[:3]):
            num_header = tuple(int(p) for p in hdr_parts[:3])

        pat = re.compile(r'(\d+)\s+(\d+)\s+(\d+)\s*\[\s*([0-9.eE+\-]+)\s*,\s*([0-9.eE+\-]+)\s*\]')
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                raise ValueError(f"Bad transition line (expect 's a s' [l,u]'): {line[:120]}")
            s_orig = int(m.group(1))
            a = str(int(m.group(2)))
            sp_orig = int(m.group(3))
            l = float(m.group(4))
            u = float(m.group(5))

            s = remap[s_orig]
            sp = remap[sp_orig]
            # trans[(s, a)].append((sp, l, u))
            # trans[(s, a)][sp] = (l, u)
            trans[(s, a)] = {sp: (l, u)}

            actions[s].add(a)

    return trans, actions, num_header

# ---------- glue ----------
def build_imdp(sta_path: str, lab_path: str, tra_path: str) -> IMDP:
    ids, remap = load_sta(sta_path)
    M = IMDP(n_states=len(ids))
    M.intervals, M.actions, hdr = load_tra_intervals(tra_path, remap)

    M.goal, M.avoid, M.init = load_lab(lab_path, remap)
    
    # Optional: sanity prints (match header stats)
    # hdr[1] == number of (s,a) pairs; hdr[2] == total transitions
    # print("Header:", hdr, " | pairs:", len(M.trans), " | edges:", sum(len(v) for v in M.trans.values()))
    return M





M = build_imdp(
    "MDPs/Ab_UAV_10-10-2025_15-15-51/Ab_UAV_10-10-2025_15-15-51/N=20000_0/Abstraction_interval.sta",
    "MDPs/Ab_UAV_10-10-2025_15-15-51/Ab_UAV_10-10-2025_15-15-51/N=20000_0/Abstraction_interval.lab",
    "MDPs/Ab_UAV_10-10-2025_15-15-51/Ab_UAV_10-10-2025_15-15-51/N=20000_0/Abstraction_interval.tra"
)
