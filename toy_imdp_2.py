from collections import defaultdict
from typing import Dict, Set, Tuple, FrozenSet

State = int
Action = str
Label = FrozenSet[str]


class ToyIMDP:
    def __init__(self):
        self.states: Set[State] = {0, 1}

        # same shape as your imdp.py
        self.actions: Dict[State, Set[Action]] = defaultdict(set)
        self.intervals: Dict[Tuple[State, Action], Dict[State, Tuple[float, float]]] = {}
        self.label: Dict[State, Label] = {}

        # labels
        self.label[0] = frozenset({"g", "init"})
        self.label[1] = frozenset({"b"})

        # actions
        self.actions[0].add("a0")
        self.actions[0].add("a3")
        self.actions[1].add("a1")

        # transitions: exact probabilities written as intervals [p,p]
        self.intervals[(0, "a0")] = {
            1: (0.5, 0.5),
            0: (0.5, 0.5),
        }

        self.intervals[(0, "a3")] = {
            0: (1, 1),
        }

        self.intervals[(1, "a1")] = {
            1: (0.5, 0.5),
            0: (0.5, 0.5),
        }


# usage
if __name__ == "__main__":
    I = ToyIMDP()

    print("states =", I.states)
    print("actions =", dict(I.actions))
    print("labels =", I.label)
    print("intervals =")
    for k, v in I.intervals.items():
        print("  ", k, "->", v)