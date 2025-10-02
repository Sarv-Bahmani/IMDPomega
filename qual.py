def qualitative_buchi_mdp(Product):
    return {
        "product_states": Product.states,
        "AECs": Product.aecs,
        "target_union": Product.target,
        "winning_product_states": Product.win_region,
    }


