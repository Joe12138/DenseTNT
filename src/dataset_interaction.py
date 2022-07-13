import sys
sys.path.append("/home/joe/Desktop/DenseTNT")

import torch
import os
import pickle
import math
import json

import argparse
import logging

import numpy as np
import utils
import zlib

from tqdm import tqdm
from typing import Dict, Optional, List
from hdmap.hd_map import HDMap
from hdmap.util.map_util import get_lane_id_in_xy_bbox, get_polygon, find_local_lane_centerlines
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from src.utils import get_name, get_angle, rotate, larger, get_pad_vector, assert_, get_subdivide_points, get_dis

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05
max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

def get_sub_map(args: utils.Args, x, y, scene_name, case_id, track_id, hd_map: HDMap, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.

    Only take lanes which are no more than args.max_distance away from (x, y).

    """
    if args.not_use_api:
        pass
    else:
        if "semantic_lane" in args.other_params:
            lane_ids = get_lane_id_in_xy_bbox(
                query_x=x,
                query_y=y,
                hd_map=hd_map,
                query_search_range_manhattan=args.max_distance
            )
            local_lane_centerlines = [hd_map.id_lane_dict[lane_id].centerline_array for lane_id in lane_ids]
            polygons = local_lane_centerlines

            if args.visualize:
                angle = mapping["angle"]
                vis_lanes = [get_polygon(hd_map.id_lane_dict[lane_id]) for lane_id in lane_ids]
                t = []
                for each in vis_lanes:
                    for point in each:
                        point[0], point[1] = rotate(point[0]-x, point[1]-y, angle)
                    num = len(each) // 2
                    t.append(each[:num].copy())
                    t.append(each[num:num*2].copy())
                vis_lanes = t
                mapping["vis_lanes"] = vis_lanes
        else:
            polygons = find_local_lane_centerlines(
                query_x=x,
                query_y=y,
                hd_map=hd_map,
                query_search_range_manhattan=args.max_distance
            )
        polygons = [polygon[:,:2].copy() for polygon in polygons]
        angle = mapping['angle']

        for index_polygon, polygon in enumerate(polygons):
            for i, point in enumerate(polygon):
                point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                if 'scale' in mapping:
                    assert 'enhance_rep_4' in args.other_params
                    scale = mapping['scale']
                    point[0] *= scale
                    point[1] *= scale
        if args.use_centerline:
            if 'semantic_lane' in args.other_params:
                local_lane_centerlines = [polygon for polygon in polygons]

        def dis_2(point):
            return point[0] * point[0] + point[1] * point[1]

        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

        def get_dis_for_points(point, polygon):
            dis = np.min(np.square(polygon[:, 0] - point[0]) + np.square(polygon[:, 1] - point[1]))
            return np.sqrt(dis)

        def ok_dis_between_points(points, points_, limit):
            dis = np.inf
            for point in points:
                dis = np.fmin(dis, get_dis_for_points(point, points_))
                if dis < limit:
                    return True
            return False

        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

        lane_idx_2_polygon_idx = {}
        for polygon_idx, lane_idx in enumerate(lane_ids):
            lane_idx_2_polygon_idx[lane_idx] = polygon_idx

        if 'goals_2D' in args.other_params:
            points = []
            visit = {}
            point_idx_2_unit_vector = []

            mapping['polygons'] = polygons

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                if 'subdivide' in args.other_params:
                    subdivide_points = get_subdivide_points(polygon)
                    points.extend(subdivide_points)
                    subdivide_points = get_subdivide_points(polygon, include_self=True)

            mapping['goals_2D'] = np.array(points)

        for index_polygon, polygon in enumerate(polygons):
            # assert_(2 <= len(polygon) <= 10, info=len(polygon))
            # assert len(polygon) % 2 == 1

            # if args.visualize:
            #     traj = np.zeros((len(polygon), 2))
            #     for i, point in enumerate(polygon):
            #         traj[i, 0], traj[i, 1] = point[0], point[1]
            #     mapping['trajs'].append(traj)

            start = len(vectors)
            if 'semantic_lane' in args.other_params:
                assert len(lane_ids) == len(polygons)
                lane_id = lane_ids[index_polygon]
                lane_segment = hd_map.id_lane_dict[lane_id]
            assert_(len(polygon) >= 2)

            lane_id = lane_ids[index_polygon]
            has_control, turn_direction, is_intersect, speed_limit = hd_map.get_lane_info(lane_id)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * args.hidden_size
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                    vector[-5] = 1
                    vector[-6] = i

                    vector[-7] = len(polyline_spans)

                    if 'semantic_lane' in args.other_params:
                        vector[-8] = 1 if has_control else -1
                        vector[-9] = 1 if turn_direction == 'RIGHT' else \
                            -1 if turn_direction == 'LEFT' else 0
                        vector[-10] = 1 if is_intersect else -1
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    if i >= 2:
                        point_pre_pre = polygon[i - 2]
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
                point_pre = point

            end = len(vectors)
            if start < end:
                polyline_spans.append([start, end])
    return (vectors, polyline_spans)


def preprocess(args, id2info, mapping, scene_name, case_id, track_id, hd_map: HDMap):
    """
    This function calculates matrix based on information from get_instance.
    """
    polyline_spans = []
    keys = list(id2info.keys())
    keys.remove(track_id)
    keys = [track_id]+keys

    vectors = []
    two_seconds = mapping['two_seconds']
    mapping["trajs"] = []
    mapping["agents"] = []

    for id in keys:
        polyline = {}

        info = id2info[id]
        start = len(vectors)

        if args.no_agents:
            if id != track_id:
                break
        
        agent = []
        for i, line in enumerate(info):
            if larger(line[DATA_DICT["frame_id"]], two_seconds):
                break
            agent.append((line[DATA_DICT["x"]], line[DATA_DICT["y"]]))
        
        if args.visualize:
            traj = np.zeros([args.hidden_size])
            for i, line in enumerate(info):
                if larger(line[DATA_DICT["frame_id"]], two_seconds):
                    traj = traj[:i*2].copy()
                    break
                traj[i*2], traj[i*2+1] = line[DATA_DICT["x"]], line[DATA_DICT["y"]]
                if i == len(info)-1:
                    traj = traj[:(i+1)*2].copy()
            traj = traj.reshape((-1, 2))
            mapping["trajs"].append(traj)

        for i, line in enumerate(info):
            if larger(line[DATA_DICT["frame_id"]], two_seconds):
                break
            x, y = line[DATA_DICT["x"]], line[DATA_DICT["y"]]
            if i > 0:
                vector = [line_pre[DATA_DICT["x"]], line_pre[DATA_DICT["y"]], x, y, line[DATA_DICT["frame_id"]], False,
                line[DATA_DICT["track_id"]]==track_id, line[DATA_DICT["track_id"]]!=track_id, len(polyline_spans), i]
                vectors.append(get_pad_vector(vector))
            line_pre = line
        
        end = len(vectors)
        if end-start == 0:
            assert id != track_id
        else:
            mapping['agents'].append(np.array(agent))
            polyline_spans.append([start, end])
    assert_(len(mapping["agents"])==len(polyline_spans))
    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping["map_start_polyline_idx"] = len(polyline_spans)
    if args.use_map:
        vectors, polyline_spans = get_sub_map(
            args=args,
            x=mapping["cent_x"],
            y=mapping["cent_y"],
            scene_name=scene_name,
            case_id=case_id,
            track_id=track_id,
            hd_map=hd_map,
            vectors=vectors,
            polyline_spans=polyline_spans,
            mapping=mapping
        )

    matrix = np.array(vectors)
    labels = []
    info = id2info[track_id]

    info = info[mapping['agent_pred_index']:]
    if not args.do_test:
        if 'set_predict' in args.other_params:
            pass
        else:
            assert len(info) == 30
    for line in info:
        labels.append(line[DATA_DICT["x"]])
        labels.append(line[DATA_DICT["y"]])
    
    if 'set_predict' in args.other_params:
        if 'test' in args.data_dir[0]:
            labels = [0.0 for _ in range(60)]

    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-2:])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'stage_one' in args.other_params:
            stage_one_label = 0
            polygons = mapping['polygons']
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    stage_one_label = i

            mapping['stage_one_label'] = stage_one_label

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([30, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
    ))

    return mapping

def argoverse_get_instance(case_array: np.ndarray, scene_name: str, case_id: int, track_id: int, args, hd_map: HDMap):
    """
    Extract polylines from one example file content
    """
    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping["file_name"] = f"{scene_name}_{case_id}_{track_id}"

    for i in range(case_array.shape[0]):
        line = case_array[i, :]

        if i == 0:
            mapping["start_time"] = int(line[DATA_DICT["frame_id"]])
            mapping["city_name"] = scene_name

        line[DATA_DICT["frame_id"]] = int(line[DATA_DICT["frame_id"]])-mapping["start_time"]
        line[DATA_DICT["x"]] = float(line[DATA_DICT["x"]])
        line[DATA_DICT["y"]] = float(line[DATA_DICT["y"]])
        id = line[DATA_DICT["track_id"]]

        # if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
        #     line[TRACK_ID] = line[OBJECT_TYPE]

        if line[DATA_DICT["track_id"]] in id2info:
            id2info[line[DATA_DICT["track_id"]]].append(line)
            vector_num += 1
        else:
            id2info[line[DATA_DICT["track_id"]]] = [line]
        
        if line[DATA_DICT["track_id"]] == track_id and len(id2info[track_id]) == 10:
            assert "cent_x" not in mapping

            agent_lines = id2info[track_id]
            mapping["cent_x"] = agent_lines[-1][DATA_DICT["x"]]
            mapping["cent_y"] = agent_lines[-1][DATA_DICT["y"]]
            mapping["agent_pred_index"] = len(agent_lines)
            mapping["two_seconds"] = line[DATA_DICT["frame_id"]]
            if "direction" in args.other_params:
                span = agent_lines[-6]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][DATA_DICT["x"]] - span[j][DATA_DICT["x"]], span[j + interval][DATA_DICT["y"]] - span[j][DATA_DICT["y"]]
                            angles.append([der_x, der_y])
            der_x, der_y = agent_lines[-1][DATA_DICT["x"]]-agent_lines[-1][DATA_DICT["x"]], agent_lines[-1][DATA_DICT["y"]]-agent_lines[-2][DATA_DICT["y"]]
    if not args.do_test:
        if "set_predict" in args.other_params:
            pass
        else:
            assert len(id2info[track_id]) == 40
    
    if vector_num > max_vector_num:
        max_vector_num = vector_num
    
    if "cent_x" not in mapping:
        return None

    if args.do_eval:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info[track_id][10:]):
            origin_labels[i][0], origin_labels[i][1] = line[DATA_DICT["x"]], line[DATA_DICT["y"]]
        mapping["origin_labels"] = origin_labels
    
    angle = -get_angle(der_x, der_y)+math.radians(90)
    if "direction" in args.other_params:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping["angle"] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[DATA_DICT["x"]], line[DATA_DICT["y"]] = rotate(line[DATA_DICT["x"]] - mapping['cent_x'], line[DATA_DICT["y"]] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[DATA_DICT["x"]] *= scale
            line[DATA_DICT["y"]] *= scale
    return preprocess(
        args=args,
        id2info=id2info,
        mapping=mapping,
        scene_name=scene_name,
        case_id=case_id,
        track_id=track_id,
        hd_map=hd_map
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        args, 
        batch_size, 
        data_path: str,
        map_path: str,
        target_veh_path: str,
        mode: str,
        save_dir: str,
        to_screen=True) -> None:
        # data prepare
        self.target_veh_list, self.scene_set = self.get_target_veh_list(target_veh_path)
        self.map_dict = self.get_map_dict(map_path)
        self.data_dict = self.get_data_dict(data_path, mode)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # data_dir = args.data_dir
        self.ex_list = []
        self.args = args

        file_list = os.listdir(self.save_dir)
        if len(file_list) == 0:
            for scene_name, case_id, track_id in tqdm(self.target_veh_list):
                case_df = self.data_dict[scene_name].get_case_data(case_id)
                instance = argoverse_get_instance(
                    case_array=case_df.values,
                    scene_name=scene_name,
                    case_id=case_id,
                    track_id=track_id,
                    args=self.args,
                    hd_map=self.map_dict[scene_name]
                )
                if instance is not None:
                    f = open(os.path.join(self.save_dir, f"{scene_name}_{case_id}_{track_id}.pkl"), "wb")
                    pickle.dump(pickle.dumps(instance), f)
                    f.close()
    def __len__(self):
        return len(self.target_veh_list)

    def __getitem__(self, idx):
        scene_name, case_id, track_id = self.target_veh_list[idx]
        f = open(os.path.join(self.save_dir, f"{scene_name}_{case_id}_{track_id}.pkl"), "rb")
        instance = pickle.load(f)
        f.close()
        instance = pickle.loads(instance)
        return instance
    
    def get_target_veh_list(self, target_veh_path: str):
        file_list = os.listdir(target_veh_path)

        scene_set = set()
        target_veh_list = []
        for file_name in file_list:
            scene_name = file_name[:-5]
            scene_set.add(scene_name)

            with open(os.path.join(target_veh_path, file_name), "r", encoding="UTF-8") as f:
                target_dict = json.load(f)

                for k in target_dict.keys():
                    case_id = int(k)

                    for track_id in target_dict[k]:
                        target_veh_list.append((scene_name, case_id, track_id))
                f.close()

            # Debug
            break
        return target_veh_list, scene_set

    def get_map_dict(self, map_path: str) -> Dict[str, HDMap]:
        map_dict = {}

        for scene in self.scene_set:
            hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene}.osm"))
            map_dict[scene] = hd_map
        
        return map_dict

    def get_data_dict(self, data_path: str, mode: str) -> Dict[str, DatasetPandas]:
        data_dict = {}

        for scene in self.scene_set:
            data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene}_{mode}.csv"))
            data_dict[scene] = data_pandas
        
        return data_dict

def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde

def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.
    The Brier Score is defined here:
        Brier, G. W. Verification of forecasts expressed in terms of probability. Monthly weather review, 1950.
        https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR, brier-minADE, brier-minFDE
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade, brier_min_ade = [], [], []
    min_fde, prob_min_fde, brier_min_fde = [], [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(pruned_trajectories[min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade
            )
            brier_min_ade.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_ade)
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde
            )
            brier_min_fde.append((1 - pruned_probabilities[min_idx]) ** 2 + curr_min_fde)

    metric_results["minADE"] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)
    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
        metric_results["brier-minADE"] = sum(brier_min_ade) / len(brier_min_ade)
        metric_results["brier-minFDE"] = sum(brier_min_fde) / len(brier_min_fde)
    return metric_results

def post_eval(args, file2pred, file2labels, DEs):

    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
        # if 'minFDE' in args.other_params:
        #     score_file += '.minFDE'
    if args.method_span[0] >= utils.NMS_START:
        score_file += '.NMS'
    else:
        score_file += '.score'

    for method in utils.method2FDEs:
        FDEs = utils.method2FDEs[method]
        miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)
        if method >= utils.NMS_START:
            method = 'NMS=' + str(utils.NMS_LIST[method - utils.NMS_START])
        utils.logging(
            'method {}, FDE {}, MR {}, other_errors {}'.format(method, np.mean(FDEs), miss_rate, utils.other_errors_to_string()),
            type=score_file, to_screen=True, append_time=True)
    utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                  type=score_file, to_screen=True, append_time=True)
    metric_results = get_displacement_errors_and_miss_rate(file2pred, file2labels, 6, 30, 2.0)
    utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score,
                      type=score_file, to_screen=True, append_time=True)

    utils.logging(vars(args), is_json=True,
                  type=score_file, to_screen=True, append_time=True)



if __name__ == '__main__':
    path_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2"
    mode = "val"
    data_path = os.path.join(path_prefix, mode)
    map_path = os.path.join(path_prefix, "maps")
    target_veh_path = os.path.join(path_prefix, f"{mode}_target_filter")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    save_dir = "/home/joe/Desktop/trained_model/dense_tnt/test_output"

    os.makedirs(save_dir, exist_ok=True)
    
    print("here")

    data_set = Dataset(
        args=args,
        batch_size=2,
        data_path=data_path,
        map_path=map_path,
        target_veh_path=target_veh_path,
        mode=mode,
        save_dir=save_dir
    )

    instance = data_set.__getitem__(1)
    instance = pickle.loads(instance)
    print("hello")



    

