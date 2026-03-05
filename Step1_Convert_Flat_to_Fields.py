#!/usr/bin/env python
# coding: utf-8

# # File Tweaks Professor Made in Advance
#
# Converted to CSV
#
#
# ## Found that some column names were zero padded on Month:
# ```
# ---------------------------------------------------------------------------
# KeyError                                  Traceback (most recent call last)
# <ipython-input-22-50d2ee988660> in <module>
#      39             for f in feats: # Combine with features
#      40                 data_key = "_".join([f,b,t])  # build up the key
# ---> 41                 data = row[data_key]
#      42                 #print("{} = {}".format(data_key,data))
#      43                 row_data.append(data)
#
# KeyError: 'mean_B1_1988_4'
# ```
# ### Confirmed with Linux tools
#
# ```
# [scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_09'
# Burn_1995_09
# mean_B1_1995_09
# stdev_B1_1995_09
# min_B1_1995_09
# ```
#
# ### Tested a SED fix on _07
#
# ```
# [scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07' | sed -e 's/_07/_7/g'
# Burn_2009_7
# mean_B1_2009_7
# stdev_B1_2009_7
# ```
#
# ### Applied the fix
#
# ```
# [scottgs@ballast BurntEnds]$ sed -i -e 's/_07/_7/g' Burn_FinalTable.csv
# [scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07'
# [scottgs@ballast BurntEnds]$ sed -i -e 's/_09/_9/g' -e 's/_06/_6/g' -e 's/_06/_6/g' -e 's/_05/_5/g' -e 's/_04/_4/g'  Burn_FinalTable.csv
#
# ```

import csv

import pandas as pd

DATAFILE_WIDE = "/dsa/data/geospatial/Prairie/Burn_FinalTable.csv"


def Get_Years_Months_From_Header(header):
    """
    Years and Months look like 'Burn_1983_8'
    This function filters to columns that start with "Burn_" and pulls out
    the YYYY_M key
    """
    burn_ym_prefix = "Burn_"
    years_months = []
    for c in header:  # for each header column
        if c.startswith(burn_ym_prefix):  # if a Burn_*
            years_months.append(
                c[len(burn_ym_prefix) :]
            )  # get the characters after Burn_

    return years_months  # send back this list


def BurntEnds_to_Sandwich(row, years_mo, bands, feats):

    object_id = row["\ufeffOBJECTID"]
    print(object_id)  #

    ###########################
    # Each Year becomes a pseudo-image row
    ###########################
    df_rows = []
    for t in years_mo:
        # print("*"*20, t)
        isBurnt = row["Burn_" + t]
        # print("Burnt {}".format(isBurnt))

        # First Column is the Year_Mo
        row_data = [t]

        # Walk the Bands (B1-B6)
        for b in bands:
            for f in feats:  # Combine with features
                data_key = "_".join([f, b, t])  # build up the key
                data = row[data_key]
                # print("{} = {}".format(data_key,data))
                row_data.append(data)

        # Last Column is the Burn
        row_data.append(isBurnt)

        # print(row_data)
        df_rows.append(row_data)

    # All years processed, create a data fram
    df = pd.DataFrame(df_rows)
    cols = ["YrMo"]
    for b in bands:
        for f in feats:  # Combine with features
            col_name = "_".join([f, b])  # build up the key
            cols.append(col_name)
    cols.append("isBurnt")
    df.columns = cols
    df.to_csv("plots/pseudo_image_{}.csv".format(object_id), index=False, header=True)

    return True


bands = ["B1", "B2", "B3", "B4", "B5", "B6"]
feats = ["mean", "stdev", "min", "max"]


# ## Test one Row

# OBJECTID *	Shape *	Burn_1983_8	mean_B1_1983_8
#  1			Polygon	1			1712.5


with open(DATAFILE_WIDE, "r") as read_obj:
    csv_reader = csv.DictReader(read_obj)

    ###########################
    # Preprocess the Header
    ###########################

    column_names = csv_reader.fieldnames
    # print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...

    years_mo = Get_Years_Months_From_Header(column_names)  # see function above
    # print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...

    ###########################
    # Now read a row of data
    ###########################
    row = next(csv_reader)
    # print(row) # OrderedDict([('\ufeffOBJECTID', '1'),
    # ('Shape *', 'Polygon'), ('Burn_1983_8', '1'),
    # ('mean_B1_1983_8', '1712.5'), ...

    BurntEnds_to_Sandwich(row, years_mo, bands, feats)


#    header = next(csv_reader)
#    print(header)


# # Chew through the data

bands = ["B1", "B2", "B3", "B4", "B5", "B6"]
feats = ["mean", "stdev", "min", "max"]

with open(DATAFILE_WIDE, "r") as read_obj:
    csv_reader = csv.DictReader(read_obj)

    ###########################
    # Preprocess the Header
    ###########################

    column_names = csv_reader.fieldnames
    # print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...

    years_mo = Get_Years_Months_From_Header(column_names)  # see function above
    # print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...

    ###########################
    # For each row, write out the file.
    ###########################
    for row in csv_reader:
        # print(row) # OrderedDict([('\ufeffOBJECTID', '1'),
        # ('Shape *', 'Polygon'), ('Burn_1983_8', '1'),
        # ('mean_B1_1983_8', '1712.5'), ...
        BurntEnds_to_Sandwich(row, years_mo, bands, feats)


# # The data has been split into fields (polygons/plots)
#
# ## The sub-folder `plots` has a all the data.
#
# Example `plots/pseudo_image_2646.csv`
#
# ```
# YrMo,mean_B1,stdev_B1,min_B1,max_B1,mean_B2,stdev_B2,min_B2,max_B2,mean_B3,stdev_B3,min_B3,max_B3,mean_B4,stdev_B4,min_B4,max_B4,mean_B5,stdev_B5,min_B5,max_B5,mean_B6,stdev_B6,min_B6,max_B6,isBurnt
# 1983_8,2171.5,3885.317078,0,8940,2427.25,4342.109664,0,9814,2348.5,4202.212132,0,9674,4131,7436.119696,0,17790,3423.8125,6125.547477,0,13945,2692.625,4824.232994,0,11611,1
# 1984_9,2300.0625,4114.751632,0,9309,2553.25,4567.609696,0,10335,2631.3125,4707.677003,0,10703,3757.125,6729.286251,0,15757,3893.125,6971.439802,0,16244,3113.375,5571.094673,0,12884,1
# 1985_4,2199.3125,3934.839323,0,9009,2422.625,4334.572672,0,9962,2375.6875,4252.603449,0,10006,4209.3125,7549.136434,0,17656,3527.1875,6310.554769,0,14454,2740,4905.740542,0,11516,1
# ```
