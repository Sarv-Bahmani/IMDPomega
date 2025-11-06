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



import csv
import matplotlib.pyplot as plt
from pathlib import Path


def plot_x(results, x_var, y_var, pic_name, x_lab, unit=1):
    """Plot function as provided"""
    results.sort(key=lambda d: d[x_var])
    xs = [d[x_var]/unit for d in results]
    ys = [d[y_var] for d in results]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(x_lab)
    plt.ylabel(y_var)
    plt.grid(True)
    plt.savefig(f"{pic_name}.png")
    plt.close()


def generate_all_plots(csv_path):
    csv_path = Path(csv_path)
    
    data = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            record['Transitions'] = int(row['Transitions'])
            record['Exported States (PRISM)'] = int(row['Exported States (PRISM)'])
            record['VI_Execution_time_sec'] = float(row['VI_Execution_time_sec'])
            record['Convergence_iteration'] = int(row['Convergence_iteration'])
            record['Qualitative_time_sec'] = float(row['Qualitative_time_sec'])
            data.append(record)
    
    # Plot 1: Transitions vs VI_Execution_time_sec
    plot_x(data, 'Transitions', 'VI_Execution_time_sec', 
           'Transitions_vs_VI_Execution_time', 'Transitions')
    
    # Plot 2: Transitions vs Convergence_iteration
    plot_x(data, 'Transitions', 'Convergence_iteration', 
           'Transitions_vs_Convergence_iteration', 'Transitions')
    
    # Plot 3: Transitions vs Qualitative_time_sec
    plot_x(data, 'Transitions', 'Qualitative_time_sec', 
            'Transitions_vs_Qualitative_time', 'Transitions')
    
    # Plot 4: Exported States (PRISM) vs VI_Execution_time_sec
    plot_x(data, 'Exported States (PRISM)', 'VI_Execution_time_sec', 
           'ExportedStates_vs_VI_Execution_time', 'Exported States (PRISM)')
    
    # Plot 5: Exported States (PRISM) vs Convergence_iteration
    plot_x(data, 'Exported States (PRISM)', 'Convergence_iteration', 
           'ExportedStates_vs_Convergence_iteration', 'Exported States (PRISM)')
    
    # Plot 6: Exported States (PRISM) vs Qualitative_time_sec
    plot_x(data, 'Exported States (PRISM)', 'Qualitative_time_sec', 
            'ExportedStates_vs_Qualitative_time', 'Exported States (PRISM)')




if __name__ == "__main__":
    adds = [
    'Ab_UAV_10-16-2025_20-48-14',
    # 'Ab_UAV_10-16-2025_13-57-21',
    # 'Ab_UAV_10-16-2025_15-11-36',
    # 'Ab_UAV_10-16-2025_15-16-07',
    # 'Ab_UAV_10-16-2025_15-25-59',
    # 'Ab_UAV_10-16-2025_15-29-37'
    ]
    for add in adds:
        print(f"Will Process IMDP at address: {add}")
        I = IMDP(address=add, noise_samples=noise_samples)
        print("\tIMDP is loaded.")
        all_labsets = {I.label[s] for s in I.states}
        B = Automata(all_labsets, "my_automaton.hoa", read_from_hoa=True)
        print("\tWill build product...")
        P = Product(I, B)
        print("\tProduct is built.")





        # # print('product is build')
        # print("Number of product states:", len(P.states))
        # common_init_target = P.init_states & P.target
        # common_init_losing = P.init_states & P.losing_sink
        # common_init_losing_or_ddlck = len({
        # x for x in P.init_states 
        # if P.imdp.label.get(x[0], frozenset()) & frozenset({"failed", "deadlock"})})
        # # common_target_losing = P.target & P.losing_sink

        # print("Init ∩ Target:", len(common_init_target))
        # print("Init ∩ Losing:", len(common_init_losing))
        # print("Init ∩ (Losing ∪ Deadlock):", common_init_losing_or_ddlck)
        # # print("Target ∩ Losing:", len(common_target_losing))
        # print("target states:", len(P.target))
        # print("losing states:", len(P.losing_sink))

        # only_init = P.init_states - (P.target | P.losing_sink)
        # print("only init:", len(only_init))

        # all_inits = len(P.init_states)
        # print("all inits:", all_inits)






        results = {}
        print("\t\tWill run value iteration...")
        results_val_iter = value_iteration_scope(P, eps=1e-9)
        plot_init_evolution_val_iter(results_val_iter, add)
        print("\t\tValue iteration is done.")

        results_strtgy = {}

        results.update({"Qualitative_time_sec": P.qualitative_time_sec})
        results.update(results_val_iter)
        results.update(results_strtgy)

        update_csv_reslt(csv_path, add, results)

    generate_all_plots(csv_path)


    



