from elements import BuchiA, MDP, Product
from qual import qualitative_buchi_mdp

if __name__ == "__main__":
    MDP = MDP()
    s0, s1, s2, s3 = 0, 1, 2, 3
    MDP.states.update([s0, s1, s2, s3])

    MDP.actions[s0].add("a")
    MDP.actions[s1].update(["safe", "risky", "detour"])
    MDP.actions[s2].update(["safe", "loop"])
    MDP.actions[s3].add("a")

    MDP.trans_MDP[(s0, "a")]      = {s0: 0.6, s1: 0.4}
    MDP.trans_MDP[(s1, "safe")]   = {s0: 1.0}
    MDP.trans_MDP[(s1, "risky")]  = {s3: 1.0}
    MDP.trans_MDP[(s1, "detour")] = {s2: 1.0}
    MDP.trans_MDP[(s2, "safe")]   = {s0: 1.0}
    MDP.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    MDP.trans_MDP[(s3, "a")]      = {s3: 1.0}

    MDP.label[s0] = frozenset({"g"})
    MDP.label[s1] = frozenset({"b"})
    MDP.label[s2] = frozenset({"b"})
    MDP.label[s3] = frozenset({"b"})

    AP = {"g", "b"}
    Buchi = BuchiA(AP) # GF g
    Buchi.add_state(0, initial=True, accepting=False)   # q0
    Buchi.add_state(1, accepting=True)                  # q1
    all_labels = {MDP.label[s0], MDP.label[s1], MDP.label[s2], MDP.label[s3]}
    for lab in all_labels:
        if "g" in lab:
            Buchi.add_edge(0, lab, 1)
            Buchi.add_edge(1, lab, 1)
        else:
            Buchi.add_edge(0, lab, 0)
            Buchi.add_edge(1, lab, 0)



    Prod = Product(MDP, Buchi)
    res = qualitative_buchi_mdp(Prod)
    print("#product states:", len(res["product_states"]))
    print("#AECs:", len(res["AECs"]))
    print("winning |W|:", len(res["winning_product_states"]))
    winners_base = {s for (s, q) in res["winning_product_states"]}
    print("base winners:", winners_base)
