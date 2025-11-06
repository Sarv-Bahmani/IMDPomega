import csv
from pathlib import Path
import matplotlib.pyplot as plt


from imdp import IMDP
from automata import Automata
from product import Product

from value_iteration import value_iteration_scope, plot_init_evolution_val_iter

noise_samples=20000
csv_path = Path("gen_imdp_info/IMDPs_info.csv")
root_models = Path("MDPs")


def update_csv_reslt(csv_path, address, res):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    row_found = False
    for row in rows:
        if row["address"].strip() == address:
            row["Execution_time_sec"] = f"{res['Execution_time_sec']:.6f}"
            row["Convergence_iteration"] = str(res["Convergence_iteration"])
            row["Qualitative_time_sec"] = str(res.get("Qualitative_time_sec", ""))
            row_found = True
            break

    if not row_found:
        row = {fn: "" for fn in fieldnames}
        row["address"] = address
        row["Noise Samples"] = str(20000)
        row["Execution_time_sec"] = f"{res['Execution_time_sec']:.6f}"
        row["Convergence_iteration"] = str(res["Convergence_iteration"])
        row["Qualitative_time_sec"] = str(res.get("Qualitative_time_sec", ""))
        rows.append(row)

    with csv_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_x(results, x_var, y_var, pic_name, x_lab, unit=1):
    results.sort(key=lambda d: d[x_var])
    xs = [d[x_var]/unit for d in results]
    ys = [d[y_var] for d in results]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_lab)
    plt.grid(True)
    plt.savefig(f"{pic_name}.png")


if __name__ == "__main__":
    adds = [
    'Ab_UAV_10-16-2025_20-48-14',
    'Ab_UAV_10-16-2025_13-57-21',
    'Ab_UAV_10-16-2025_15-11-36',
    'Ab_UAV_10-16-2025_15-16-07',
    'Ab_UAV_10-16-2025_15-25-59',
    'Ab_UAV_10-16-2025_15-29-37'
    ]
    # for add in adds:
    add = adds[0]



    I = IMDP(address=add, noise_samples=noise_samples)
    all_labsets = {I.label[s] for s in I.states}
    B = Automata(all_labsets, "my_automaton.hoa", read_from_hoa=True)
    P = Product(I, B)





    # print('product is build')
    print("Number of product states:", len(P.states))
    common_init_target = P.init_states & P.target
    common_init_losing = P.init_states & P.losing_sink
    common_init_losing_or_ddlck = len({
    x for x in P.init_states 
    if P.imdp.label.get(x[0], frozenset()) & frozenset({"failed", "deadlock"})})
    # common_target_losing = P.target & P.losing_sink

    print("Init ∩ Target:", len(common_init_target))
    print("Init ∩ Losing:", len(common_init_losing))
    print("Init ∩ (Losing ∪ Deadlock):", common_init_losing_or_ddlck)
    # print("Target ∩ Losing:", len(common_target_losing))
    print("target states:", len(P.target))
    print("losing states:", len(P.losing_sink))

    only_init = P.init_states - (P.target | P.losing_sink)
    print("only init:", len(only_init))

    all_inits = len(P.init_states)
    print("all inits:", all_inits)






    results = {}
    
    results_val_iter = value_iteration_scope(P, eps=1e-9)
    plot_init_evolution_val_iter(results_val_iter, add)

    results_strtgy = {}

    results.update({"Qualitative_time_sec": P.qualitative_time_sec})
    results.update(results_val_iter)
    results.update(results_strtgy)

    update_csv_reslt(csv_path, add, results)
