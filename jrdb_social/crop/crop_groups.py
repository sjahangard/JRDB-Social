from dataclasses import dataclass
import json
from multiprocessing import cpu_count
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorneko_util as N
from tqdm.auto import tqdm

from ..data import test_seqs, train_seqs, valid_seqs
from ..misc import parse_multi_simin_dict, parse_simin_dict

pattern = re.compile(r"(.+?):([123])\((\d+?)\.jpg,(\d+?)\.jpg\),?")


@dataclass
class GroupGlobalEntry:
    video_name: str
    group_id: int
    main_location: str
    main_location_difficulty: int
    intention: str
    intention_difficulty: int

    # key: (video_name, group_id), value: List[GroupGlobalEntry]
    entries = dict()

    def __hash__(self) -> int:
        return hash((self.video_name, self.group_id, self.main_location, self.main_location_difficulty, self.intention, self.intention_difficulty))

    @classmethod
    def add(
        cls, video_name: str, group_id: int, current_timestamp: int,
        main_location: str, main_location_difficulty: int,
        intention: str, intention_difficulty: int
    ):
        if (video_name, group_id) in cls.entries:
            entry = cls.entries[(video_name, group_id)]
            assert entry.main_location == main_location and entry.main_location_difficulty == main_location_difficulty \
                and entry.intention == intention and entry.intention_difficulty == intention_difficulty, \
                (entry, main_location, main_location_difficulty,
                 intention, intention_difficulty)
        else:
            cls.entries[(video_name, group_id)] = cls(video_name, group_id,
                                                      main_location, main_location_difficulty, intention, intention_difficulty)

    @classmethod
    def clear(cls):
        cls.entries = dict()


@dataclass
class GroupLocalLocationEntry:
    video_name: str
    group_id: int
    start: int
    end: Optional[int]
    local_location: str
    local_location_difficulty: int

    # key: (video_name, group_id, start, end, local_location, local_location_difficulty), value: List[GroupLocalLocationEntry]
    entries = dict()
    prev_frame_entries = set()
    current_frame_entries = set()

    def __hash__(self) -> int:
        return hash((self.video_name, self.group_id, self.start, self.end,
                     self.local_location, self.local_location_difficulty))

    @classmethod
    def add(
        cls, video_name: str, group_id: int, current_timestamp: int,
        local_location: str, local_location_difficulty: int
    ):

        for entry in cls.prev_frame_entries:
            if entry.video_name == video_name and entry.group_id == group_id \
                    and entry.local_location == local_location and entry.local_location_difficulty == local_location_difficulty:

                entry.end = current_timestamp
                cls.current_frame_entries.add(entry)
                return
        else:
            cls.current_frame_entries.add(cls(video_name, group_id, current_timestamp, current_timestamp,
                                          local_location, local_location_difficulty))

    @classmethod
    def flush(cls):
        # find these entries which is in the previous frame but not in the current frame
        for entry in cls.prev_frame_entries:
            if entry not in cls.current_frame_entries and entry.local_location != "unknown":
                cls.entries[(entry.video_name, entry.group_id, entry.start, entry.end,
                                entry.local_location, entry.local_location_difficulty)] = entry
        cls.prev_frame_entries = cls.current_frame_entries
        cls.current_frame_entries = set()

    @classmethod
    def clear(cls):
        cls.prev_frame_entries = set()
        cls.current_frame_entries = set()
        cls.entries = dict()


@dataclass
class GroupSalientEntry:
    video_name: str
    group_id: int
    start: int
    end: Optional[int]
    salient: str
    salient_difficulty: int

    entries = dict()
    prev_frame_entries = set()
    current_frame_entries = set()

    def __hash__(self) -> int:
        return hash((self.video_name, self.group_id, self.start, self.end, self.salient, self.salient_difficulty))

    @classmethod
    def add(cls, video_name: str, group_id: int, current_timestamp: int, salient: str, salient_difficulty: int):
        for entry in cls.prev_frame_entries:
            if entry.video_name == video_name and entry.group_id == group_id \
                and entry.salient == salient and entry.salient_difficulty == salient_difficulty:
                entry.end = current_timestamp
                cls.current_frame_entries.add(entry)
                return
        else:
            cls.current_frame_entries.add(cls(video_name, group_id, current_timestamp, current_timestamp, salient, salient_difficulty))

    @classmethod
    def flush(cls):
        for entry in cls.prev_frame_entries:
            if entry not in cls.current_frame_entries and entry.salient != "unknown":
                cls.entries[(entry.video_name, entry.group_id, entry.start, entry.end, entry.salient, entry.salient_difficulty)] = entry
        cls.prev_frame_entries = cls.current_frame_entries
        cls.current_frame_entries = set()

    @classmethod
    def clear(cls):
        cls.prev_frame_entries = set()
        cls.current_frame_entries = set()
        cls.entries = dict()


def parse_local_location(text):
    assert "[" in text and "]" in text, text
    text = text.lstrip("[").rstrip("]")
    if text == "unknown":
        return {"type": "unknown"}
    elif "," in text:
        parse_results = pattern.findall(text)
        assert len(parse_results) > 0, text
        return {
            "type": "multiple",
            "labels": [
                {
                    "timestamps": sorted([int(start), int(end)]),
                    "local_location": label,
                    "difficulty": difficulty
                }
                for label, difficulty, start, end in parse_results
            ]
        }
    else:
        label, *difficulty = text.split(":")
        return {"type": "single", "local_location": label, "difficulty": difficulty[0] if len(difficulty) > 0 else None}


def process_bbox(bboxes, padding_scale):
    bboxes = np.array(bboxes)
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    x1 = bboxes[:, 0].min()
    y1 = bboxes[:, 1].min()
    x2 = bboxes[:, 2].max()
    y2 = bboxes[:, 3].max()

    x = (x1 + x2) // 2
    y = (y1 + y2) // 2

    w = x2 - x1
    h = y2 - y1

    margin = max(w, h) * padding_scale // 2

    x1, x2 = x - margin, x + margin
    y1, y2 = y - margin, y + margin
    return x1, x2, y1, y2


def crop_group_video(group_data, video_name, group_id, groups_out_folder, local_locations_out_folder, data_root, padding_scale=2.5):
    # save as video
    video_writer = cv2.VideoWriter(
        f"{groups_out_folder}/{group_id}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512))

    local_location_video_writers = dict()
    if group_data["local_location"]["type"] == "multiple":
        for entry in group_data["local_location"]["labels"]:
            start, end = entry["timestamps"]
            local_location_video_writers[f"{start}-{end}"] = (start, end, cv2.VideoWriter(
                f"{local_locations_out_folder}/{group_id}_{start}_{end}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512)))
        for entry in group_data["salient"]["labels"]:
            start, end = entry["timestamps"]
            if f"{start}-{end}" not in local_location_video_writers:
                local_location_video_writers[f"{start}-{end}"] = (start, end, cv2.VideoWriter(
                    f"{local_locations_out_folder}/{group_id}_{start}_{end}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512)))

    frame_names = sorted(group_data["box"].keys())

    for frame_name in frame_names:
        bboxes = group_data["box"][frame_name]
        frame = cv2.imread(
            f"{data_root}/images/image_stitched/{video_name}/{frame_name:06}.jpg")
        _, total_w, _ = frame.shape
        # expand the frame to the right
        frame = np.concatenate([frame, frame], axis=1)
        # find the x coordinate of the persons
        person_xs = [x1 for x1, _, _, _ in bboxes]
        person_xs = np.array(person_xs) % total_w
        # calculate the each pair of person's distance
        # and then check if any of them is larger than the threshold (half of the total width)
        person_pair_distances = np.abs(person_xs[None] - person_xs[None].T)
        if np.any(person_pair_distances > (total_w / 2)):
            index_persons_in_left = np.where(person_xs < total_w / 2)[0]
            # if there are persons in the left, then add the total width to their x coordinate, to make it continuous
            for i in index_persons_in_left:
                bboxes[i][0] += total_w

        x1, x2, y1, y2 = process_bbox(bboxes, padding_scale)
        if x1 < 0:
            x1 += total_w
            x2 += total_w
        frame = N.preprocess.crop_with_padding(
            frame, int(x1), int(x2), int(y1), int(y2), 0)
        frame = cv2.resize(frame, (512, 512))
        video_writer.write(frame)

        for start, end, writer in local_location_video_writers.values():
            if start <= int(frame_name) <= end:
                writer.write(frame)

    video_writer.release()


def main(data_root, split):

    if split == "train":
        seqs = train_seqs
    elif split == "valid":
        seqs = valid_seqs
    elif split == "test":
        seqs = test_seqs
    else:
        raise ValueError(f"Unknown split: {split}")

    # metadata = pd.read_csv("../JRDB-Social-Original/all_labels_group.txt", sep=" ", names=[
    #                        "video", "group", "local_location", "intention", "main_location"])

    for video_name in seqs:
        all_groups = defaultdict(dict)  # key: cluster_ID, value: frame_names
        labels = N.io.read.json(f"{data_root}/labels/labels_2d_activity_social_stitched/{video_name}.json")["labels"]
        all_frame_names = sorted(labels.keys())

        # build the segment based group labels
        for frame_name in all_frame_names:
            current_timestamp = int(frame_name[:-4])
            for entry in labels[frame_name]:
                for group_info in entry["group_info"]:
                    main_location, main_location_difficulty = parse_simin_dict(
                        group_info["venue"])
                    intention, intention_difficulty = parse_simin_dict(
                        group_info["aim"])
                    local_location, local_location_difficulty = parse_simin_dict(
                        group_info["BPC"])
                    salient, salient_difficulty = parse_multi_simin_dict(group_info["SSC"])
                    
                    GroupLocalLocationEntry.add(video_name, entry["social_group"]["cluster_ID"], current_timestamp,
                                      local_location, local_location_difficulty)
                    GroupSalientEntry.add(video_name, entry["social_group"]["cluster_ID"], current_timestamp,
                                        salient, salient_difficulty)
                    GroupGlobalEntry.add(video_name, entry["social_group"]["cluster_ID"], current_timestamp,
                                            main_location, main_location_difficulty, intention, intention_difficulty)
            
            GroupLocalLocationEntry.flush()
            GroupSalientEntry.flush()
        GroupLocalLocationEntry.flush()
        GroupSalientEntry.flush()

        # get bbox for each frame
        for frame_name in all_frame_names:
            for entry in labels[frame_name]:
                if "box" not in all_groups[entry["social_group"]["cluster_ID"]]:
                    all_groups[entry["social_group"]
                               ["cluster_ID"]]["box"] = defaultdict(list)
                all_groups[entry["social_group"]["cluster_ID"]]["box"][int(frame_name.replace(".jpg", ""))].append(entry["box"])

        del_keys = []
        for key in all_groups:
            all_groups[key]["box"] = dict(all_groups[key]["box"])

            # global attr
            global_entry = GroupGlobalEntry.entries.get((video_name, key), None)
            if global_entry is None:
                del_keys.append(key)
                continue

            all_groups[key]["main_location"] = {
                "label": global_entry.main_location,
                "difficulty": global_entry.main_location_difficulty
            }

            all_groups[key]["intention"] = {
                "label": global_entry.intention,
                "difficulty": global_entry.intention_difficulty
            }

            # local location
            local_location_entries = list(
                (entry for entry in GroupLocalLocationEntry.entries.values() if entry.video_name == video_name and entry.group_id == key)
            )

            all_groups[key]["local_location"] = {
                "type": "multiple",
                "labels": [
                    {
                        "timestamps": [entry.start, entry.end],
                        "label": entry.local_location,
                        "difficulty": entry.local_location_difficulty
                    }
                    for entry in local_location_entries
                ]
            }

            # salient
            salient_entries = list(
                (entry for entry in GroupSalientEntry.entries.values() if entry.video_name == video_name and entry.group_id == key)
            )

            all_groups[key]["salient"] = {
                "type": "multiple",
                "labels": [
                    {
                        "timestamps": [entry.start, entry.end],
                        "label": entry.salient,
                        "difficulty": entry.salient_difficulty
                    }
                    for entry in salient_entries
                ]
            }

        for key in del_keys:
            del all_groups[key]

        # remove groups with no frame
        for key in all_groups:
            # count the number of frames for each local_location segments
            # remove the segments with no frame
            if all_groups[key]["local_location"]["type"] == "multiple":
                keep_index = []
                for i, entry in enumerate(all_groups[key]["local_location"]["labels"]):
                    start, end = entry["timestamps"]
                    n_frames = len([frame_name for frame_name in all_groups[key]["box"].keys(
                    ) if start <= frame_name <= end])
                    if n_frames > 0:
                        keep_index.append(i)

                all_groups[key]["local_location"]["labels"] = [
                    all_groups[key]["local_location"]["labels"][i] for i in keep_index]
                
            # salient
            if all_groups[key]["salient"]["type"] == "multiple":
                keep_index = []
                for i, entry in enumerate(all_groups[key]["salient"]["labels"]):
                    start, end = entry["timestamps"]
                    n_frames = len([frame_name for frame_name in all_groups[key]["box"].keys(
                    ) if start <= frame_name <= end])
                    if n_frames > 0:
                        keep_index.append(i)

                all_groups[key]["salient"]["labels"] = [
                    all_groups[key]["salient"]["labels"][i] for i in keep_index]

        (groups_out_folder := Path(
            f"{data_root}/cropped/groups/{video_name}")).mkdir(parents=True, exist_ok=True)
        (local_locations_out_folder := Path(
            f"{data_root}/cropped/local_locations/{video_name}")).mkdir(parents=True, exist_ok=True)

        for group_id in tqdm(all_groups.keys()):
            crop_group_video(all_groups[group_id], video_name, group_id, groups_out_folder, local_locations_out_folder, data_root)

        # with ProcessPoolExecutor(cpu_count()) as executor:
        #     futures = []

        #     for group_id in all_groups.keys():
        #         futures.append(executor.submit(
        #             crop_group_video, all_groups[group_id], video_name, group_id, groups_out_folder, local_locations_out_folder, data_root))

        #     for future in tqdm(futures, desc=f"{video_name}"):
        #         future.result()

        N.io.write.json(f"{groups_out_folder}/labels.json", dict(all_groups), fast=False)
