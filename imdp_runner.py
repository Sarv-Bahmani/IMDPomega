import csv
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from imdp import IMDP
from automata import Automata
from product import Product

from value_iteration import value_iteration_scope, plot_init_evolution_val_iter
from strategy_gurobi import strategy_improve_scope, plot_init_evolution_stra_impr

noise_samples=20000
noise_samples_str = "Noise Samples"
csv_path = Path("gen_imdp_info/IMDPs_info.csv")
root_models = Path("MDPs")





address_str = "address"
val_iter_time_str = "Val_Iter_Execution_time_sec"
val_iter_converge_iter_str = "Val_Iter_Convergence_iteration"
qual_time_str = "Qualitative_time_sec"
transitions_str = 'Transitions'
Exported_States_PRISM_str = 'Exported States (PRISM)'
Choices_str = "Choices"

strat_imprv_Values_str = "Stratgy_Imprv_Values"
strat_imprv_Convergence_iteration_str = "Stratgy_Imprv_Convergence_iteration"
strat_imprv_Execution_time_sec_str = "Stratgy_Imprv_Execution_time_sec"
mean_V_list_str = "mean_V_list"

ratio_str = "ratio_SI_VI"

def update_row(row, res):
    row[qual_time_str] = str(res.get(qual_time_str, ""))

    row[val_iter_time_str] = f"{res[val_iter_time_str]:.6f}"
    row[val_iter_converge_iter_str] = str(res[val_iter_converge_iter_str])
    
    row[strat_imprv_Execution_time_sec_str] = str(res[strat_imprv_Execution_time_sec_str])
    row[strat_imprv_Convergence_iteration_str] = str(res[strat_imprv_Convergence_iteration_str])


        

def update_csv_reslt(csv_path, address, res):
    rows = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    row_found = False
    for row in rows:
        if row[address_str].strip() == address:
            update_row(row, res)
            row_found = True
            break

    if not row_found:
        row = {fn: "" for fn in fieldnames}
        row[address_str] = address
        row[noise_samples_str] = str(20000)
        update_row(row, res)
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
    plt.ylabel(y_var)
    plt.grid(True)
    plt.savefig(f"{pic_name}.png")
    plt.close()



def plot_ratio_scatter(data):
    models = [rec['model'] for rec in data]
    ratios = [rec['ratio_SI_VI'] for rec in data]

    plt.figure(figsize=(len(models)/2,3))
    # plt.bar(models, ratios)
    x = np.arange(len(models)) 
    plt.bar(x, ratios) 
    plt.axhline(1.0, linestyle='--', linewidth=1)
    plt.ylabel('SI_time / VI_time')
    # plt.xticks(rotation=0, ha='right', fontsize=8)
    plt.xticks(x, models, rotation=45, ha='right', fontsize=8)
    plt.title('Relative Cost: Strategy Improvement vs Value Iteration')
    plt.tight_layout()
    plt.savefig("ratio_SI_VI_bar_chart.png", dpi=500)
    plt.close()



def generate_all_plots(csv_path):
    csv_path = Path(csv_path)
    
    data = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            record[transitions_str] = int(row[transitions_str])
            record[Exported_States_PRISM_str] = int(row[Exported_States_PRISM_str])
            # record[Choices_str] = int(row[Choices_str])

            record[qual_time_str] = float(row[qual_time_str])
            
            record[val_iter_time_str] = float(row[val_iter_time_str])
            record[val_iter_converge_iter_str] = int(row[val_iter_converge_iter_str])
            
            record[strat_imprv_Execution_time_sec_str] = float(row[strat_imprv_Execution_time_sec_str])

            record[ratio_str] = float(record[strat_imprv_Execution_time_sec_str] / record[val_iter_time_str])

            record['model'] = row[address_str][:4]

            data.append(record)
    


    plot_ratio_scatter(data)
    

    x_var_list = [transitions_str, Exported_States_PRISM_str]
    y_var_list = [qual_time_str, val_iter_time_str,  val_iter_converge_iter_str, strat_imprv_Execution_time_sec_str, ratio_str]
    for x_var in x_var_list:
        for y_var in y_var_list:

            plot_x(data, x_var, y_var, f"{y_var}_vs_{x_var}", x_var)


if __name__ == "__main__":
    adds = [
        # "784-0.5-Ab_UAV_10-16-2025_20-48-14",
        # "1024-1-Ab_UAV_11-14-2025_07-27-06",
        # "1225-2-Ab_UAV_11-14-2025_07-25-03",
        # "1296-2-Ab_UAV_11-14-2025_07-33-15",
        # "1600-3-Ab_UAV_10-16-2025_13-57-21",
        # "1800-5-Ab_UAV_10-16-2025_15-11-36",
        # "2025-9-Ab_UAV_11-14-2025_07-35-35",
        # "2160-9-Ab_UAV_10-16-2025_15-16-07",
        # "2304-12-Ab_UAV_11-14-2025_08-04-41",
        "2430-15-Ab_UAV_10-16-2025_15-25-59",
        # "2916-25-Ab_UAV_10-16-2025_15-29-37"
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
        eps = 1e-9



        print("\t\tWill run value iteration...")
        results_val_iter = value_iteration_scope(P, eps)
        plot_init_evolution_val_iter(results_val_iter, add[:7])
        print("\t\tValue iteration is done.")


        print("\t\tWill run strategy improve ...")
        results_strtgy = strategy_improve_scope(P, eps)
        plot_init_evolution_stra_impr(results_strtgy, add[:7])
        print("\t\tstrategy improve is done.")

        results.update({qual_time_str: P.qualitative_time_sec})
        results.update(results_val_iter)
        results.update(results_strtgy)

        pd.DataFrame.from_dict(results, orient="index").to_csv(f"results_{add[:7]}.csv")


        print(f"\tUpdating results to CSV...")
        update_csv_reslt(csv_path, add, results)
        print(f"\tCSV is updated.")





    generate_all_plots(csv_path)


    



