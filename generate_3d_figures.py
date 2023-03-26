import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import warnings

from mpl_toolkits.mplot3d import Axes3D

warnings.simplefilter(action='ignore', category=FutureWarning)


dataset = "ROSMAP"
exp_path = "exp_hyperparams"
missing_rate = 0.2
lambda_al = 0.1

df_results = pd.read_csv(f"{exp_path}/best_results_{dataset}.csv", sep="\t")
df_results = df_results[df_results["missing_rate"]==missing_rate]
df_results = df_results[df_results['lambda_al']==lambda_al]
df_results = df_results[df_results['lambda_cil']>0.]
df_results = df_results.reset_index(drop=True)
df_results


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_x = len(np.unique(df_results['lambda_cil']))
n_y = len(np.unique(df_results['lambda_co']))
print(n_x)
print(n_y)

colors = ["#352A86", "#217ECF", "#38B89C", "#D4BC52", "#F7FA0D"][::-1]
for idx_y in range(n_y):
    xpos = np.arange(n_x)
    ypos = np.zeros(n_x)+idx_y
    zpos = np.zeros(n_x)
    dx = np.ones(n_x)*0.8
    dy = np.ones(n_x)*0.8
    sub_df = df_results[df_results['lambda_co']==np.unique(df_results['lambda_co'])[idx_y]][::-1]
    dz = sub_df['acc'].to_list()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors[idx_y], alpha=1, shade=False, edgecolor = "black")
    
ax.view_init(25, 45)
ax.set_xlabel(r"$\lambda_{cl}$")
ax.set_xticks(np.arange(n_x))
ax.set_xticklabels(np.unique(df_results['lambda_cil']))
print(np.unique(df_results['lambda_cil']))

ax.set_ylabel(r"$\lambda_{co}$")
ax.set_yticks(np.arange(n_y))
ax.set_yticklabels(np.unique(df_results['lambda_co']))
print(np.unique(df_results['lambda_co']))

ax.set_zlabel(r"$Accuracy$")

ax.set_title(r"Missing Rate=0.2, $\lambda_{al}$=0.1")
fig.tight_layout()
plt.show()