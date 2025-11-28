import os
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE_FONTSIZE = 10

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",

    "font.size": BASE_FONTSIZE,      
    "axes.labelsize": BASE_FONTSIZE+1, 
    "axes.titlesize": BASE_FONTSIZE+1,
    "xtick.labelsize": BASE_FONTSIZE,    
    "ytick.labelsize": BASE_FONTSIZE,
    "legend.fontsize": BASE_FONTSIZE,
})




import sys
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imdp import IMDP
from automata import Automata
from product import Product

from value_iteration import value_iteration_scope, plot_init_evolution_val_iter
from strategy_gurobi import strategy_improve_scope, plot_init_evolution_stra_impr

noise_samples=20000
noise_samples_str = "Noise Samples"
root_models = Path("MDPs")

shuttle_str = "shuttle"
uav_str = "uav"


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



# def plot_x(results, x_var, y_var, pic_name, x_lab, unit=1):
#     results.sort(key=lambda d: d[x_var])
#     xs = [d[x_var]/unit for d in results]
#     ys = [d[y_var] for d in results]
#     plt.figure()
#     plt.plot(xs, ys, marker="o")
#     plt.xlabel(x_lab)
#     plt.ylabel(y_var)
#     plt.grid(True)
#     plt.savefig(os.path.join("results", "plots", f"{pic_name}.png"))
#     plt.close()


def plot_x(results, x_var, y_var, pic_name, x_lab, unit=1, y_label=None, line_label=None, figsize=(5.2, 3.2)):
    results.sort(key=lambda d: d[x_var])
    xs = [d[x_var] / unit for d in results]
    ys = [d[y_var] for d in results]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(xs, ys, marker="o", linewidth=2.0,
            markersize=4, label=line_label)

    ax.set_xlabel(x_lab)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if line_label is not None:
        leg = ax.legend(frameon=True)
        leg.get_frame().set_linewidth(1.0)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(BASE_FONTSIZE)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(BASE_FONTSIZE)

    fig.tight_layout()
    out_dir = os.path.join("results", "plots")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{pic_name}.png"), dpi=500)

    plt.close(fig)




def plot_ratio_scatter(data):
    models = [rec['model'] for rec in data]
    ratios = [rec['ratio_SI_VI'] for rec in data]

    plt.figure(figsize=(len(models)/1.5,3.2))
    x = np.arange(len(models)) 
    plt.bar(x, ratios) 
    plt.axhline(1.0, linestyle='--', linewidth=1)
    # plt.ylabel('SI_time / VI_time')
    plt.xticks(x, models, rotation=45, ha='right', fontsize=BASE_FONTSIZE)
    # plt.title('Relative Cost: Strategy Improvement vs Value Iteration')

    plt.tight_layout()
    plt.savefig(os.path.join("results", "plots", "ratio_SI_VI_bar_chart.png"), dpi=500)

    plt.close()



def generate_all_plots(csv_path):
    csv_path = Path(csv_path)
    
    data = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            record[transitions_str] = float(row[transitions_str]) / (10**6)  # in million
            record[Exported_States_PRISM_str] = int(row[Exported_States_PRISM_str])
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
    # for x_var in x_var_list:
    for y_var in y_var_list:
        x_var = transitions_str
        plot_x(data, x_var, y_var, f"{y_var}_vs_{x_var}", f"million {transitions_str}")
        x_var = Exported_States_PRISM_str
        plot_x(data, x_var, y_var, f"{y_var}_vs_{x_var}", "States")


if __name__ == "__main__":

    if len(sys.argv) < 3:
            print("Usage: python imdp_runner.py <Model Type> <json Address>")
            sys.exit(1)

    model_type = sys.argv[1]
    csv_path = Path(f"gen_imdp_info/IMDPs_info_{model_type}.csv")

    json_path = Path(sys.argv[2])
    with json_path.open('r') as f:
        adds = json.load(f)



    for add in adds[model_type]:
        print(f"Will Process IMDP at address: {add}")
        I = IMDP(model_type=model_type, address=add, noise_samples=noise_samples)
        print("\tIMDP is loaded.")
        all_labsets = {I.label[s] for s in I.states}
        B = Automata(all_labsets, "my_automaton.hoa", read_from_hoa=True)
        print("\tWill build product...")
        P = Product(I, B)
        print("\tProduct is built.")

        results = {}

        print("\t\tWill run value iteration...")
        up_contrac_fctr = 0.999 if model_type == uav_str else 0.99
        results_val_iter = value_iteration_scope(P, up_contrac_fctr)
        plot_init_evolution_val_iter(results_val_iter, add[:14])
        print("\t\tValue iteration is done.")


        print("\t\tWill run strategy improve ...")
        results_strtgy = strategy_improve_scope(P)
        plot_init_evolution_stra_impr(results_strtgy, add[:14])
        print("\t\tstrategy improve is done.")

        results.update({qual_time_str: P.qualitative_time_sec})
        results.update(results_val_iter)
        results.update(results_strtgy)

        pd.DataFrame.from_dict(results, orient="index").to_csv(os.path.join("results", "each_imdp_result", f"results_{add[:14]}.csv"))

        print(f"\tUpdating results to CSV...")

        update_csv_reslt(csv_path, add, results)
        print(f"\tCSV is updated.")


    generate_all_plots(csv_path)


    



