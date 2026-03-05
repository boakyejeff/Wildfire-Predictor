#!/usr/bin/env python
# coding: utf-8

# # Prairie Burn Detection with Multitemporal Landsat TM
#
# Machine learning has been around for decades, but deep learning is the new focus of study within machine learning.
# The goals of implementing deep learning into remote sensing are resulting in much faster and accurate results for much larger amounts of data.
# The field of remote sensing has focused on increasing the accuracy of land classification and change detection.
# A possible solution for increasing accuracy is the use of a deep learning network that have been producing greater accuracy in image classifying and change detection.
# This study focuses on classification and change detection within the [Konza Prairie](https://en.wikipedia.org/wiki/Konza_Prairie_Biological_Station) in Geary County, Kansas.
# The images are from **Landsat4** and **Landsat5** spanning the years 1985-2011 and are trained and tested through a deep learning network.
# The experimental results show the possibility of a deep learning network producing results that could then be implement for much larger regions.
#
#
#  + Shape Length: 480 m
#  + Shape Area: 14400 m
#  + Shape Side: 120 m
#  + Pixel Size = 30x30 m  or 15x15 m
#  + 4x4 or   8x8 area?

# ## Discussion of Data Collection from ArcGIS
# ### Image to Wide Table
# #### Wide Row is one field
# #### 6 Bands

# ## Data Carpentry
#
#
# ### Wide Table conversion to Fields in Rows of Years
#
#  * requires: `/dsa/data/geospatial/Prairie/Burn_FinalTable.csv`
#  * requires: `./plots` subfolder.
#
# [Convert Flat to Fields](./Convert_Flat_to_Fields.ipynb)
#
#  * produces: `./plots/pseudo_image_N.csv`,
#    * where `N` is the ObjectID from `Burn_FinalTable`
#

#
# ### Wide Table conversion to Stacked Data
# ##### Prep for statistical analysis and visualization along bands
# ##### Prep for row-level ML baselines
#
#  * requires: `/dsa/data/geospatial/Prairie/Burn_FinalTable.csv`
#
# [Convert Flat to Stacked](./Convert_Flat_to_Stacked.ipynb)
#
#  * produces: `./FullStacked_data.csv`
#

# ## Descriptive statistical analysis and visualization along bands
#
# ### Also generates the normalized data in the [0,1] Range
#
#  * requires: `./FullStacked_data.csv`
#
# [Stacked to Visualizations](./Stacked_to_Visualizations.ipynb)
#
#  * produces: `./normalized_global_bands_mean_stdev.csv`
#
#
# ## Geospatial Visualization of Plots
#
#  * requires: `./FullStacked_data.csv`
#  * requires: `/dsa/data/geospatial/Prairie/tm_stats.gdb`
#
# [Visualize GeoDB](./Visualize_GeoDB.ipynb)
#

# ---
#
# ## Basic ML using Row-Records (Baseline)
#
# #### Raw Features
#
#  * requires: `./FullStacked_data.csv`
#
# [Stacked_BaseLine_ML](./Stacked_BaseLine_ML.ipynb)

#
# #### Normalized [0,1] Features
#
#  * requires: `./normalized_global_bands_mean_stdev.csv`
#
# [NormalizedData_BaseLine_ML](./NormalizedData_BaseLine_ML.ipynb)

# #### Standard Scaler (z-score) Features
#
#  * requires: `./FullStacked_data.csv`
#
# [Z-Score_Scaled_BaseLine_ML](./Z-Score_Scaled_BaseLine_ML.ipynb)

#
