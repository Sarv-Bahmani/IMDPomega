from matplotlib import pyplot as plt
import csv
import ast

import os

iter_init_save = 1

csv_path = os.path.join("results", "each_imdp_result", "results_1600-3-.csv")

keys_to_read = ["mean_V_list", "mean_L_list", "mean_U_list"]
data = {}

with open(csv_path, mode='r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0] in keys_to_read:
            data[row[0]] = ast.literal_eval(row[1])


mean_L_list = data["mean_L_list"]
mean_U_list = data["mean_U_list"]
mean_V_list = data["mean_V_list"]

# extend_mean_v = mean_V_list + [mean_V_list[-1]] * (len(mean_L_list) - len(mean_V_list))
# mean_V_list = extend_mean_v
mean_V_list_extend = [mean_V_list[-1]] * ((len(mean_L_list) - len(mean_V_list))+1)


x_values = list(range(iter_init_save, (len(mean_L_list)+1) * iter_init_save, iter_init_save))

plt.figure(figsize=(8/1.5,3.2))
plt.plot(x_values, mean_L_list, marker='o', label='VI Lower values')
plt.plot(x_values, mean_U_list, marker='s', label='VI Upper values')
plt.plot(x_values[:len(mean_V_list)], mean_V_list, marker="x", label='SI values', color='#C20078')
# plt.plot(x_values[len(mean_V_list)-1:], mean_V_list_extend, color='#C20078', linestyle='--')


plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"Evolution_InitSt.png")
plt.close()