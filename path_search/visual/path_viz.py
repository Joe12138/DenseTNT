import copy
from typing import List, Dict

import numpy as np
from shapely.geometry import LineString
import matplotlib.axes
from hdmap.object.lane import Lane


def plot_path(path: List[int], id_lane_dict: Dict[int, Lane], axes: matplotlib.axes, color: str = "orange"):
    for i in range(len(path)-1):
        s_id, e_id = path[i], path[i+1]

        s_lane_obj, e_lane_obj = id_lane_dict[s_id], id_lane_dict[e_id]
        s_cl_ls, e_cl_ls = LineString(s_lane_obj.centerline_array), LineString(e_lane_obj.centerline_array)

        s_p, e_p = s_cl_ls.interpolate(s_cl_ls.length/2), e_cl_ls.interpolate(e_cl_ls.length/2)

        axes.annotate("", xy=(e_p.x, e_p.y), xytext=(s_p.x, s_p.y),
                      arrowprops=dict(arrowstyle="->", color=color, linestyle="-", linewidth=2, zorder=100), alpha=0.7)

    return axes


def plot_cl(path: List[int], id_lane_dict: Dict[int, Lane], axes:matplotlib.axes, color: str = "gray"):
    cl_array = None
    for i in range(len(path)):
        s_id = path[i]
        s_lane_obj = id_lane_dict[s_id]
        if i == 0:
            cl_array = copy.deepcopy(s_lane_obj.centerline_array)
        else:
            cl_array = np.concatenate((cl_array, copy.deepcopy(s_lane_obj.centerline_array)), axis=0)
    axes.plot(cl_array[:, 0], cl_array[:, 1], color=color, linestyle="--")
    return axes


def plot_cl_array(cl: np.ndarray, axes: matplotlib.axes, color: str= "gray"):
    axes.plot(cl[:, 0], cl[:, 1], color=color, linestyle="--")
    return axes