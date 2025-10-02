from IMDPQualitative import qualitative_buchi_mdp, BuchiA, MDP, Product

if __name__ == "__main__":
    mdp = MDP()
    s0, s1, s2, s3 = 0, 1, 2, 3
    mdp.states.update([s0, s1, s2, s3])

    mdp.actions[s0].add("a")
    mdp.actions[s1].update(["safe", "risky", "detour"])
    mdp.actions[s2].update(["safe", "loop"])
    mdp.actions[s3].add("a")

    mdp.trans_MDP[(s0, "a")]      = {s0: 0.6, s1: 0.4}
    mdp.trans_MDP[(s1, "safe")]   = {s0: 1.0}
    mdp.trans_MDP[(s1, "risky")]  = {s3: 1.0}
    mdp.trans_MDP[(s1, "detour")] = {s2: 1.0}
    mdp.trans_MDP[(s2, "safe")]   = {s0: 1.0}
    mdp.trans_MDP[(s2, "loop")]   = {s2: 1.0}
    mdp.trans_MDP[(s3, "a")]      = {s3: 1.0}

    mdp.label[s0] = frozenset({"g"})
    mdp.label[s1] = frozenset({"b"})
    mdp.label[s2] = frozenset({"b"})
    mdp.label[s3] = frozenset({"b"})

    AP = {"g", "b"}
    buchi = BuchiA(AP) # GF g
    buchi.add_state(0, initial=True, accepting=False)   # q0
    buchi.add_state(1, accepting=True)                  # q1
    all_labels = {mdp.label[s0], mdp.label[s1], mdp.label[s2], mdp.label[s3]}
    for lab in all_labels:
        if "g" in lab:
            buchi.add_edge(0, lab, 1)
            buchi.add_edge(1, lab, 1)
        else:
            buchi.add_edge(0, lab, 0)
            buchi.add_edge(1, lab, 0)


    res = qualitative_buchi_mdp(mdp, buchi)

    print("#product states:", len(res["product_states"]))
    print("#AECs:", len(res["AECs"]))
    print("winning |W|:", len(res["winning_product_states"]))
    winners_base = {s for (s, q) in res["winning_product_states"]}
    print("base winners:", winners_base)
