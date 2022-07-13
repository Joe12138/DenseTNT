import sys

import numpy as np

sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from dataset.pandas_dataset import DatasetPandas
from hdmap.visual.map_vis import draw_lanelet_map
from hdmap.hd_map import HDMap

DATA_DICT = {
    "case_id": 0,
    "track_id": 1,
    "frame_id": 2,
    "timestamp_ms": 3,
    "agent_type": 4,
    "x": 5,
    "y": 6,
    "vx": 7,
    "vy": 8,
    "psi_rad": 9,
    "length": 10,
    "width": 11,
    "track_to_predict": 12
}

test_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/test_single-agent"
map_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps"
save_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/test_target_filter"

file_list = os.listdir(test_path)

for file_name in file_list:
    if file_name[:-8] != "DR_USA_Intersection_EP0":
        continue

    data_path = os.path.join(test_path, file_name)

    data_pandas = DatasetPandas(data_path=data_path)
    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{file_name[:-8]}.osm"))
    data_df = pd.read_csv(data_path)

    # data_df_filter = data_df[data_df["track_to_predict"] == 1]
    case_set = data_df[["case_id", "track_id"]].drop_duplicates(["case_id", "track_id"]).values.astype(int)
    # print("hello")

    axes = plt.subplot(111)

    axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)

    for i in range(case_set.shape[0]):
        if case_set[i][0] != 22:
            continue

        case_data = data_pandas.get_case_data(case_id=case_set[i][0])
        track_id = case_set[i][1]

        track_full_info = data_pandas.get_track_data(case_id=case_set[i][0], track_id=track_id)

        track_full_xy = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)

        if np.unique(track_full_info[:, DATA_DICT["track_to_predict"]]) == 1:
            axes.plot(track_full_xy[:, 0], track_full_xy[:, 1], color="purple")
            axes.scatter(track_full_xy[-1, 0], track_full_xy[-1, 1], color="purple", marker="o")
        else:
            axes.plot(track_full_xy[:, 0], track_full_xy[:, 1], color="b")
            axes.scatter(track_full_xy[-1, 0], track_full_xy[-1, 1], color="b", marker="o")

    plt.show()







