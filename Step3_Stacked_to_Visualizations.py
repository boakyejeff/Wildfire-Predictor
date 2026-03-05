#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic("matplotlib", "inline")
import seaborn as sns

print(sns.__version__)


bands = ["B1", "B2", "B3", "B4", "B5", "B6"]
feats = ["mean", "stdev", "min", "max"]
reduced_feats = ["mean", "stdev"]


# # Read in the stacked data

result = pd.read_csv("FullStacked_data.csv")


# # Generate Global_Statistics

global_stats = result.describe()


# # some basic visualizations of the value distributions

# Walk the Bands (B1-B6)

for b in bands:
    for f in reduced_feats:  # Combine with features
        data_key = "_".join([f, b])  # build up the key
        sns.distplot(result[data_key])
        plt.show()


# ---

# # Pull Apart, Burn vs Non-Burn

result.head().transpose()


result["isBurnt"].value_counts()


# # This normalizes the data to [0,1] ranges.

normalized = pd.DataFrame(
    {
        "OBJECTID": result["OBJECTID"],
        "isBurnt": result["isBurnt"],
        "YrMo": result["YrMo"],
    }
)

for b in bands:
    for f in reduced_feats:  # Combine with features
        data_key = "_".join([f, b])  # build up the key
        dk_min = global_stats[data_key]["min"]
        dk_max = global_stats[data_key]["max"]
        print("Key {} : Range ({},{})".format(data_key, dk_min, dk_max))
        normalized[data_key] = np.interp(result[data_key], (dk_min, dk_max), (0, 1))
normalized.head()


# # Write out the normalized bands

normalized.to_csv("normalized_global_bands_mean_stdev.csv")


nonburn = normalized[normalized["isBurnt"] == 1]
burn = normalized[normalized["isBurnt"] == 2]


bins = [x for x in np.arange(0, 1, 0.01)]

for b in bands:
    for f in reduced_feats:  # Combine with features
        data_key = "_".join([f, b])  # build up the key
        sns.distplot(nonburn[data_key], color="green", bins=bins)
        plt.show()
        sns.distplot(burn[data_key], color="red", bins=bins)
        plt.show()


# ---
