from elements import Product

def qualitative_buchi_mdp(MDP, BuchiA, Product):
    Prod = Product
    mecs = Prod.mec_decomposition()
    aecs = Prod.aecs_from_mecs(mecs)
    target = set().union(*aecs) if aecs else set()
    win_region = Prod.almost_sure_winning(target)
    return {
        "product_states": Prod.states,
        "AECs": aecs,
        "target_union": target,
        "winning_product_states": win_region,
    }, Prod


