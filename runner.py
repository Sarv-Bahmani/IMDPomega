from elements import BuchiA, MDP, Product
from qual import qualitative_buchi_mdp


if __name__ == "__main__":
    MDP = MDP()
    # s0, s1, s2, s3 = 0, 1, 2, 3
    # MDP.states.update([s0, s1, s2, s3])

    # MDP.actions[s0].add("a")
    # MDP.actions[s1].update(["safe", "risky", "detour"])
    # MDP.actions[s2].update(["safe", "loop"])
    # MDP.actions[s3].add("a")

    # MDP.trans_MDP[(s0, "a")]      = {s0: 0.6, s1: 0.4}
    # MDP.trans_MDP[(s1, "safe")]   = {s0: 1.0}
    # MDP.trans_MDP[(s1, "risky")]  = {s3: 1.0}
    # MDP.trans_MDP[(s1, "detour")] = {s2: 1.0}
    # MDP.trans_MDP[(s2, "safe")]   = {s0: 1.0}
    # MDP.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    # MDP.trans_MDP[(s3, "a")]      = {s3: 1.0}

    # MDP.label[s0] = frozenset({"g"})
    # MDP.label[s1] = frozenset({"b"})
    # MDP.label[s2] = frozenset({"b"})
    # MDP.label[s3] = frozenset({"b"})


    # s0, s1, s2, s3 = 0, 1, 2, 3
    # MDP.states.update([s0, s1, s2, s3])

    # MDP.actions[s0].add("a")
    # MDP.actions[s0].update(["safe"])
    # MDP.actions[s1].update(["risky"])
    # MDP.actions[s2].update(["loop"])
    # MDP.actions[s3].add("safe")


    # MDP.trans_MDP[(s0, "a")]      = {s0: 0.1, s1: 0.9}
    # MDP.trans_MDP[(s1, "risky")]  = {s0: 0.000001, s2: 1-0.000001}
    # MDP.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    # MDP.trans_MDP[(s0, "safe")]   = {s3: 1.0}
    # MDP.trans_MDP[(s3, "safe")]   = {s3: 1.0}


    # MDP.label[s0] = frozenset({"g"})
    # MDP.label[s1] = frozenset({"b"})
    # MDP.label[s2] = frozenset({"b"})
    # MDP.label[s3] = frozenset({"g"})





    s0, s1, s2, s3 = 0, 1, 2, 3
    MDP.states.update([s0, s1, s2, s3])
    MDP.actions[s0].add("a"); MDP.actions[s1].update(["safe", "risky"]); MDP.actions[s2].add("a"); MDP.actions[s0].update(["safe"]); MDP.actions[s3].add("safe")
    MDP.label[s0] = frozenset({"g"}); MDP.label[s1] = frozenset({"b"}); MDP.label[s2] = frozenset({"b"}); MDP.label[s3] = frozenset({"g"})
    MDP.trans_MDP[(s0, "a")] = {s0: 0.5, s1: 0.5}
    MDP.trans_MDP[(s1, "risky")] = {s2: 0.9 , s0: 0.2}
    MDP.trans_MDP[(s2, "a")] = {s2: 1.0}
    MDP.trans_MDP[(s0, "safe")]   = {s3: 1.0}
    MDP.trans_MDP[(s3, "safe")]   = {s3: 1.0}







    AP = {"g", "b"}
    Buchi = BuchiA(AP) # GF g
    Buchi.add_state(0, initial=True, accepting=False)   # q0
    Buchi.add_state(1, accepting=True)                  # q1
    all_labels = {MDP.label[st] for st in MDP.label}
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

    print(Prod.target)



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
    # s0, s1, s2, s3 = 0, 1, 2, 3
    # I.states.update([s0, s1, s2, s3])
    # I.actions[s0].add("a"); I.actions[s1].update(["safe", "risky"]); I.actions[s2].add("a"); I.actions[s0].update(["safe"]); I.actions[s3].add("safe")
    # I.label[s0] = frozenset({"g"}); I.label[s1] = frozenset({"b"}); I.label[s2] = frozenset({"b"}); I.label[s3] = frozenset({"g"})

    # I.intervals[(s0, "a")] = {s0: (0.5, 0.5), s1: (0.5, 0.5)}
    # I.intervals[(s1, "risky")] = {s2: (0.6, 0.9) , s0: (0.0, 0.4)}
    # I.intervals[(s2, "a")] = {s2: (1.0, 1.0)}
    # I.intervals[(s0, "safe")]   = {s3: (0.6, 0.8) , s1: (0.0, 0.4)}
    # I.intervals[(s3, "safe")]   = {s3: (1.0, 1.0)}






