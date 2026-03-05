#!/usr/bin/env python
# coding: utf-8

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")


fiona.listlayers("/dsa/data/geospatial/Prairie/tm_stats.gdb")


plots_gdf = gpd.read_file(
    "/dsa/data/geospatial/Prairie/tm_stats.gdb", layer="FinalTable"
)


plots_gdf.head()


plots_gdf.columns


plots_gdf.geometry.head()


slim = gpd.GeoDataFrame(
    {"OBJECTID": plots_gdf.index, "geometry": plots_gdf.geometry}, geometry="geometry"
)
slim["OBJECTID"] = slim["OBJECTID"] + 1
slim.head()


# fig, ax = plt.subplots(1, 1)

# plt.figure(figsize=(20,20))
plt.rcParams["figure.figsize"] = [12, 12]

slim.boundary.plot()


slim.to_file("Plot_Polygons")


del plots_gdf


# # Read in Stacked Data

import pandas as pd

stacked = pd.read_csv("FullStacked_data.csv")


stacked.head()


plots_gdf = slim.merge(stacked, on="OBJECTID", how="right")
plots_gdf.head()


print(plots_gdf.YrMo.unique())


# Prior to running this step make sure you create the YrMo_Shapes folder in your workspace!!
# It is required for storage of the burn plots for each burn


plt.rcParams["figure.figsize"] = [8, 8]

for yrmo in plots_gdf.YrMo.unique():
    print(yrmo)
    shapeName = "YrMo_Shapes/Burn_Plots_{}".format(yrmo)
    tmp = plots_gdf[plots_gdf["YrMo"] == yrmo]
    print(tmp.shape)
    print("Saving ShapeFile : '{}'".format(shapeName))
    tmp.to_file(shapeName)
    tmp.plot(column="isBurnt")
    plt.show()
