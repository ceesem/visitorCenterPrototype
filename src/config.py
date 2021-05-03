import numpy as np
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

synapse_table = "synapses_pni_2"

dendrite_color = (0.894, 0.102, 0.110)
axon_color = (0.227, 0.459, 0.718)
clrs = np.array([axon_color, dendrite_color])

base_dir = os.path.dirname(__file__)
data_path = f"{base_dir}/data"
ct_base_filename = f"{data_path}/minnie_cell_types_model_v83.pkl"
ct_base_df = pd.read_pickle(ct_base_filename)
ct_base_df["soma_y_um"] = ct_base_df["soma_y"] / 1000

ct_col = "cell_type_pred"
soma_depth_col = "soma_y_um"
valence_col = "is_inhib"
syn_depth_col = "syn_y_um"
soma_table = "nucleus_neuron_svm"

inhib_types = ["BC", "MC", "BPC", "NGC"]
exc_types = ["23P", "4P", "5P_IT", "5P_NP", "5P_PT", "6CT", "6IT"]
ct_base_df[valence_col] = ct_base_df[ct_col].apply(lambda x: x in inhib_types)

layer_bnds = np.load(f"{data_path}/layer_bounds_v1.npy")
height_bnds = np.load(f"{data_path}/height_bounds_v1.npy")
ticklocs = np.concatenate([height_bnds[0:1], layer_bnds, height_bnds[1:]])

e_colors = sns.color_palette("RdPu", n_colors=9)
i_colors = sns.color_palette("Greens", n_colors=9)

base_ind = 6
e_color = e_colors[base_ind]
i_color = i_colors[base_ind]

val_colors = np.array([e_color, i_color])

split_threshold = 0.7