from ast import Not
from collections import defaultdict
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import tensorneko_util as N
import torch
import torchmetrics
from torch.utils.data import Dataset


train_seqs = ['bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0',
              'clark-center-intersection-2019-02-28_0',
              'cubberly-auditorium-2019-04-22_0', 'forbes-cafe-2019-01-22_0',
              'gates-159-group-meeting-2019-04-03_0',
              'gates-basement-elevators-2019-01-17_1', 'gates-to-clark-2019-02-28_1',
              'hewlett-packard-intersection-2019-01-24_0',
              'huang-basement-2019-01-25_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0',
              'memorial-court-2019-03-16_0',
              'packard-poster-session-2019-03-20_0', 'packard-poster-session-2019-03-20_1',
              'packard-poster-session-2019-03-20_2',
              'stlc-111-2019-04-19_0', 'svl-meeting-gates-2-2019-04-08_0',
              'svl-meeting-gates-2-2019-04-08_1', 'tressider-2019-03-16_0']

valid_seqs = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0',
              'meyer-green-2019-03-16_0', 'nvidia-aud-2019-04-18_0',
              'tressider-2019-03-16_1', 'tressider-2019-04-26_2']


test_seqs = ['cubberly-auditorium-2019-04-22_1', 'discovery-walk-2019-02-28_0', 'discovery-walk-2019-02-28_1', 'food-trucks-2019-02-12_0',
             'gates-ai-lab-2019-04-17_0', 'gates-basement-elevators-2019-01-17_0', 'gates-foyer-2019-01-17_0', 'gates-to-clark-2019-02-28_0',
             'hewlett-class-2019-01-23_0', 'hewlett-class-2019-01-23_1', 'huang-2-2019-01-25_1', 'huang-intersection-2019-01-22_0',
             'indoor-coupa-cafe-2019-02-06_0', 'lomita-serra-intersection-2019-01-30_0', 'meyer-green-2019-03-16_1', 'nvidia-aud-2019-01-25_0',
             'nvidia-aud-2019-04-18_1', 'nvidia-aud-2019-04-18_2', 'outdoor-coupa-cafe-2019-02-06_0', 'quarry-road-2019-02-28_0',
             'serra-street-2019-01-30_0', 'stlc-111-2019-04-19_1', 'stlc-111-2019-04-19_2', 'tressider-2019-03-16_2', 'tressider-2019-04-26_0',
             'tressider-2019-04-26_1', 'tressider-2019-04-26_3']

ages = [
    "young_adulthood",
    "middle_adulthood",
    "late_adulthood",
    "adolescence",
    "childhood"
]

races = [
    "caucasian",
    "asians",
    "black",
    "others"
]

main_locations = ['cafeteria/dining_hall/food_court',
                  'open_area/campus',
                  'open_space/corridor',
                  'room/class',
                  'study_space',
                  'street'
                  ]

intentions = ['wandering',
              'discussing_an_object/matter',
              'studying/writing/reading/working',
              'waiting_for_someone/something',
              'socializing',
              'excursion',
              'unknown',
              'attending_class/lecture/seminar',
              'eating/ordering_food',
              'commuting',
              'navigating']


interactions = [
    'sitting together', 'waving hand together', 'walking together', 'holding sth together', 'looking at sth together', 'bending together',
    'moving together', 'going upstairs together', 'shaking hand', 'interaction with door together', 'pointing at sth together', 'standing together',
    'looking at robot together', 'walking toward each other', 'looking into sth together', 'conversation', 'eating together', 'hugging',
    'standing together together', 'going downstairs together', 'cycling together'
]

poses = ['floor', 'chair', 'stairs', 'ground', 'platform', 'grass', 'bike', 'scooter', 'balcony', 'sofa', 'skateboard', 'sidewalk',
         'pathway', 'desk', 'street', 'crosswalk', 'bench', 'road'
         ]

salients = ['counter',
            'pillar',
            'bin',
            'gate',
            'table',
            'wall',
            'stairs',
            'fence',
            'door',
            'trolley',
            'coffee_machine',
            'shelves',
            'cafeteria',
            'balcony',
            'desk',
            'room',
            'elevator',
            'sofa',
            'bike',
            'board',
            'stroller',
            'scooter',
            'statue',
            'platform',
            'standboard',
            'poster',
            'show_case',
            'window',
            'floor',
            'crutches',
            'chair',
            'tree',
            'pole',
            'bench',
            'building',
            'food_truck',
            'bus',
            'robot',
            'baggage',
            'stand_pillar',
            'screen',
            'forecourt',
            'shop',
            'cabinet',
            'light_street',
            'car',
            'copy_machine',
            'drink_fountain',
            'class'
            ]


def is_available(label):
    if label is None:
        return False
    if type(label) is dict:
        label = label["label"]
    return "impossible" not in label.lower() and "unknown" not in label.lower()


def process_main_location(label):
    return label.lower().split(":")[0].replace("indoor_", "").replace("outdoor_", "")


class JRDBSocial(Dataset):
    def __init__(self, data_root, split) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        if split == "train":
            seqs = train_seqs
        elif split == "valid":
            seqs = valid_seqs
        elif split == "test":
            seqs = test_seqs
            raise NotImplementedError
        else:
            raise ValueError

        self.seqs = seqs

    @classmethod
    def persons(cls, data_root, split):
        return JRDBSocialPersons(data_root, split)

    @classmethod
    def groups(cls, data_root, split):
        return JRDBSocialGroups(data_root, split)

    @classmethod
    def interactions(cls, data_root, split):
        return JRDBSocialInteractions(data_root, split)

    @classmethod
    def holistics(cls, data_root, split):
        return JRDBSocialHolistics(data_root, split)


class JRDBSocialPersons(JRDBSocial):

    race_map = {
        "mongoloid/asian": "asians",
        "caucasian/white": "caucasian",
        "negroid/black": "black",
        "others": "others",
        "impossible": None
    }
    
    def __init__(self, data_root, split) -> None:
        super().__init__(data_root, split)
        self.labels = {}
        for seqs in self.seqs:
            self.labels[seqs] = N.io.read.json(f"{data_root}/cropped/persons/{seqs}/labels.json")
        self.person_labels = []
        for seq, persons_data in self.labels.items():
            for person_id, person_data in persons_data.items():
                self.person_labels.append((seq, person_id, person_data))

    def __len__(self):
        return len(self.person_labels)

    def __getitem__(self, index):
        if self.split == "test":
            return NotImplementedError
        else:
            return self._getitem_train(index)

    def _getitem_train(self, index):
        video_id, person_id, person_data = self.person_labels[index]
        video_path = f"{self.data_root}/cropped/persons/{video_id}/{person_id}.avi"

        gender_label = person_data["label"]["gender"][0].lower() if person_data["label"]["gender"] is not None else None
        age_label = person_data["label"]["age"][0].lower() if person_data["label"]["age"] is not None else None
        race_label = self.race_map[person_data["label"]["race"][0].lower()] if person_data["label"]["race"] is not None else None

        gender_label = gender_label if is_available(gender_label) else None
        age_label = age_label if is_available(age_label) else None
        race_label = race_label if is_available(race_label) else None

        assert gender_label in ["male", "female", None]
        assert age_label in ages + [None], age_label
        assert race_label in races + [None]

        return video_path, gender_label, age_label, race_label


class JRDBSocialGroups(JRDBSocial):
    def __init__(self, data_root, split, scale=None) -> None:
        super().__init__(data_root, split)
        self.group_folder = "groups"
        self.local_locations_folder = "local_locations"
        self.labels = {}
        for seqs in self.seqs:
            self.labels[seqs] = N.io.read.json(f"{data_root}/cropped/groups/{seqs}/labels.json")
        self.group_labels = []
        for seq, groups_data in self.labels.items():
            for group_id, group_data in groups_data.items():
                video_path = f"{self.data_root}/cropped/{self.group_folder}/{seq}/{group_id}.avi"
                if os.path.exists(video_path):
                    self.group_labels.append(("group", video_path, group_data["intention"], group_data["main_location"]))

                # add local location video
                local_location_video_data = defaultdict(dict)
                for local_location_entry in group_data["local_location"]["labels"]:
                    start, end = sorted(local_location_entry["timestamps"])
                    v_path = f"{self.data_root}/cropped/{self.local_locations_folder}/{seq}/{group_id}_{start}_{end}.avi"
                    local_location_video_data[v_path]["local_location"] = local_location_entry
                # add salient video
                for salient_entry in group_data["salient"]["labels"]:
                    start, end = sorted(salient_entry["timestamps"])
                    v_path = f"{self.data_root}/cropped/{self.local_locations_folder}/{seq}/{group_id}_{start}_{end}.avi"
                    local_location_video_data[v_path]["salient"] = salient_entry

                # add to group labels
                for v_path, local_location_entry in local_location_video_data.items():
                    if os.path.exists(v_path):
                        self.group_labels.append(("local_location", v_path, local_location_entry.get("local_location", None), local_location_entry.get("salient", None)))

    def __len__(self):
        return len(self.group_labels)

    def __getitem__(self, index):
        if self.split == "test":
            raise NotImplementedError
        else:
            return self._getitem_train(index)

    def _getitem_train(self, index):
        datum_type, *datum = self.group_labels[index]
        if datum_type == "group":
            video_path, intention_label, main_location_label = datum
            intention_label = intention_label["label"].lower().split("&") if is_available(intention_label) else None
            main_location_label = process_main_location(main_location_label["label"]) if is_available(main_location_label) else None
            return video_path, None, None, intention_label, main_location_label
        elif datum_type == "local_location":
            video_path, local_location_label, salient_label = datum
            local_location_label = local_location_label["label"] if is_available(local_location_label) else None
            salient_label = salient_label["label"].split("&") if is_available(salient_label) else None
            # assert local_location_label is None or local_location_label in poses, local_location_label
            # assert salient_label is None or salient_label in salients, salient_label
            return video_path, local_location_label, salient_label, None, None
        else:
            raise ValueError


class JRDBSocialInteractions(JRDBSocial):
    def __init__(self, data_root, split) -> None:
        super().__init__(data_root, split)
        self.labels = {}
        for seqs in self.seqs:
            self.labels[seqs] = N.io.read.json(f"{data_root}/cropped/interactions/{seqs}/labels.json")
    
        self.interaction_labels = []
        for seq, interactions_data in self.labels.items():
            for file, interaction_data in interactions_data.items():
                self.interaction_labels.append((seq, file, interaction_data))

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, index):
        if self.split == "test":
            raise NotImplementedError
        else:
            return self._getitem_train(index)

    def _getitem_train(self, index):
        video_name, file, datum = self.interaction_labels[index]

        video_path = f"{self.data_root}/cropped/interactions/{video_name}/{file}"
        # assert os.path.exists(video_path), video_path
        interaction_label = datum["interaction"]

        return video_path, interaction_label


def iou_2d(proposal, target):
    """
    Calculate 2D IOU for M proposals with N targets.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The proposals array with shape [M, 4]. The 4
            columns represents x1, y1, x2, y2.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The targets array with shape [N, 4]. The 4 columns
            represents x1, y1, x2, y2.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.tensor(proposal)

    if type(target) is np.ndarray:
        target = torch.tensor(target)

    proposal_x1 = proposal[:, 0]
    proposal_y1 = proposal[:, 1]
    proposal_x2 = proposal[:, 2]
    proposal_y2 = proposal[:, 3]

    target_x1 = target[:, 0].unsqueeze(0).T
    target_y1 = target[:, 1].unsqueeze(0).T
    target_x2 = target[:, 2].unsqueeze(0).T
    target_y2 = target[:, 3].unsqueeze(0).T

    inner_x1 = torch.maximum(proposal_x1, target_x1)
    inner_y1 = torch.maximum(proposal_y1, target_y1)
    inner_x2 = torch.minimum(proposal_x2, target_x2)
    inner_y2 = torch.minimum(proposal_y2, target_y2)

    area_proposal = (proposal_x2 - proposal_x1) * (proposal_y2 - proposal_y1)
    area_target = (target_x2 - target_x1) * (target_y2 - target_y1)

    inter_x = torch.clamp(inner_x2 - inner_x1, min=0.)
    inter_y = torch.clamp(inner_y2 - inner_y1, min=0.)
    inter = inter_x * inter_y

    union = area_proposal + area_target - inter

    return inter / union, inter / area_proposal


class JRDBSocialHolistics(JRDBSocial):
    def __init__(self, data_root, split):
        super().__init__(data_root, split)
        persons_dataset = [*JRDBSocial.persons(data_root, split)]
        groups_dataset = [*JRDBSocial.groups(data_root, split)]
        interactions_dataset = [*JRDBSocial.interactions(data_root, split)]

        self.persons_metadata = dict()
        for video_path, gender_label, age_label, race_label in persons_dataset:
            seq, pid = video_path.split("/")[-2:]
            self.persons_metadata[(seq, pid.replace(".avi", ""))] = {
                "gender_label": gender_label,
                "age_label": age_label,
                "race_label": race_label
            }

        self.groups_metadata = dict()
        for video_path, local_location_label, salient_label, intention_label, main_location_label in groups_dataset:
            seq, gid = video_path.split("/")[-2:]
            self.groups_metadata[(seq, gid.replace(".avi", ""))] = {
                "local_location_label": local_location_label,
                "salient_label": salient_label,
                "intention_label": intention_label,
                "main_location_label": main_location_label
            }

        self.interactions_metadata = dict()
        for video_path, interaction_label in interactions_dataset:
            seq, fid = video_path.split("/")[-2:]
            gid, start, end, uid1, uid2 = fid.split("_")

            self.interactions_metadata[(seq, gid)] = self.interactions_metadata.get(
                (seq, gid), list())
            self.interactions_metadata[(seq, gid)].extend(interaction_label)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        files = [f"{self.data_root}/cropped/holistics/{seq}/{i}.avi" for i in range(5)]

        if self.split == "test":
            raise NotImplementedError

        meta = self.load_metadata_of_seq(seq)
        genders_label = [{"male": 0, "female": 0} for _ in range(5)]
        ages_label = [dict(zip(ages, [0] * len(ages))) for _ in range(5)]
        races_label = [dict(zip(races, [0] * len(races))) for _ in range(5)]

        poses_label = [dict(zip(poses, [0] * len(poses))) for _ in range(5)]
        salients_label = [dict(zip(salients, [0] * len(salients)))
                          for _ in range(5)]
        intentions_label = [
            dict(zip(intentions, [0] * len(intentions))) for _ in range(5)]

        main_locations_label = None
        interactions_label = [
            dict(zip(interactions, [0] * len(interactions))) for _ in range(5)]

        for view_index in range(5):

            for pid in meta[view_index]["persons"]:
                if (seq, pid) not in self.persons_metadata:
                    continue
                persons_meta = self.persons_metadata[(seq, pid)]
                if persons_meta["gender_label"] is not None:
                    genders_label[view_index][persons_meta["gender_label"]] += 1
                if persons_meta["age_label"] is not None:
                    ages_label[view_index][persons_meta["age_label"]] += 1
                if persons_meta["race_label"] is not None:
                    races_label[view_index][persons_meta["race_label"]] += 1

            for gid in meta[view_index]["groups"]:
                groups_meta = self.groups_metadata[(seq, gid)]
                if groups_meta["local_location_label"] is not None:
                    for video_path, local_location_label in groups_meta["local_location_label"].items():
                        if local_location_label is not None:
                            if local_location_label["pose"] is not None:
                                poses_label[view_index][local_location_label["pose"]] += 1
                            if local_location_label["salients"] is not None:
                                for salient in local_location_label["salients"]:
                                    salients_label[view_index][salient] += 1
                if groups_meta["intention_label"] is not None:
                    for intention in groups_meta["intention_label"]:
                        try:
                            intentions_label[view_index][intention] += 1
                        except KeyError:
                            pass
                if main_locations_label is None and groups_meta["main_location_label"] is not None:
                    main_locations_label = groups_meta["main_location_label"]

            for gid in meta[view_index]["groups"]:
                interactions_meta = self.interactions_metadata.get(
                    (seq, gid), list())
                for interaction in interactions_meta:
                    if interaction in interactions:
                        interactions_label[view_index][interaction] += 1

        main_locations_label = [main_locations_label] * 5
        return files, genders_label, ages_label, races_label, poses_label, salients_label, intentions_label, main_locations_label, interactions_label

    def load_metadata_of_seq(self, seq_id: str):

        view_coordinates = np.array([
            [i * 752, 0, (i + 1) * 752, 480] for i in range(5)
        ])

        persons_in_views = [[], [], [], [], []]
        groups_in_views = [[], [], [], [], []]

        persons_bboxes = N.io.read.json(os.path.join(self.data_root, "cropped", "persons", seq_id, "labels.json"))
        for person_id, person_data in persons_bboxes.items():
            box = np.array([each[1] for each in person_data["box"]])
            # (x1, y1, w, h) -> (x1, y1, x2, y2)
            box[:, 2] += box[:, 0]
            box[:, 3] += box[:, 1]

            _, ioas = iou_2d(box, view_coordinates)  # (num_frames, num_views)
            person_is_in_view = ioas.numpy().max(1) > 0.5

            for view_id, is_in_view in enumerate(person_is_in_view):
                if is_in_view:
                    persons_in_views[view_id].append(person_id)

        groups_bboxes = N.io.read.json(os.path.join(self.data_root, "cropped", "groups", seq_id, "labels.json"))
        for group_id, group_data in groups_bboxes.items():
            box = np.array([self.merge_group_bbox(each)
                           for each in group_data["box"].values()])
            # (x1, x2, y1, y2) -> (x1, y1, x2, y2)
            box = box[:, [0, 2, 1, 3]]

            _, ioas = iou_2d(box, view_coordinates)
            group_is_in_view = ioas.numpy().max(1) > 0.5

            for view_id, is_in_view in enumerate(group_is_in_view):
                if is_in_view:
                    groups_in_views[view_id].append(group_id)

        return [{
            "persons": persons_in_views[i],
            "groups": groups_in_views[i]
        } for i in range(5)]

    @staticmethod
    def merge_group_bbox(bboxes):
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

        margin = max(w, h) // 2

        x1, x2 = x - margin, x + margin
        y1, y2 = y - margin, y + margin
        return x1, x2, y1, y2


def calculate_acc_and_f1(preds, labels, categories):
    preds = torch.tensor(
        [categories.index(each) if each is not None else -1 for each in preds])
    labels = torch.tensor([categories.index(each) for each in labels])

    # calculate accuracy
    total_items = len(labels)
    correct = (preds == labels).sum().item()

    acc = correct / total_items
    print(f"Accuracy {correct} / {total_items} = {acc}")

    # calculate f1
    if len(categories) == 2:
        metric = torchmetrics.F1Score(
            task="binary", num_classes=len(categories))
        # assign the None prediction as the wrong class for easier calculation of f1
        preds[preds == -1] = 1 - labels[preds == -1]
    else:
        metric = torchmetrics.F1Score(
            task="multiclass", num_classes=len(categories), average="macro")
        # assign the None prediction as the wrong class for easier calculation of f1
        preds[preds == -1] = (labels[preds == -1] + 1) % len(categories)

    f1 = metric(preds, labels)
    print(f"F1 {f1}")
    return correct, total_items, acc, float(f1)


def calculate_acc_and_f1_for_multilabel(preds, labels, categories):
    preds = np.stack([N.util.sparse2binary(np.array([categories.index(
        each1) for each1 in each], dtype=int), length=len(categories)) for each in preds])
    labels = np.stack([N.util.sparse2binary(np.array([categories.index(
        each1) for each1 in each], dtype=int), length=len(categories)) for each in labels])

    preds = preds.flatten()
    labels = labels.flatten()

    # calculate accuracy
    total_items = len(labels)
    correct = (preds == labels).sum().item()

    acc = correct / total_items
    print(f"Accuracy {correct} / {total_items} = {acc}")

    # calculate f1
    metric = torchmetrics.F1Score(task="binary", average="micro")

    f1 = metric(torch.tensor(preds), torch.tensor(labels))
    print(f"F1 {f1}")
    return correct, total_items, acc, float(f1)


def calculate_acc_and_f1_for_holistic(records_dict):
    acc = []
    f1 = []
    for record in records_dict.values():
        for label in record["label"].keys():
            TP = min(record["answers"][label], record["label"][label])
            FP = max(record["answers"][label] - record["label"][label], 0)
            FN = max(record["label"][label] - record["answers"][label], 0)
            total = max(record["answers"][label], record["label"][label])
            if total == 0:
                continue
            acc.append(TP / total)
            f1.append(TP / (TP + 0.5 * (FP + FN)))

    acc, f1 = float(np.mean(acc)), float(np.mean(f1))
    print(f"Accuracy {acc}")
    print(f"F1 {f1}")
    return acc, f1


def calculate_acc_and_f1_for_holistic_category(records_dict):
    pred = []
    label = []
    f1 = []
    for record in records_dict.values():
        for gt in record["label"].keys():
            pred.append(record["answers"][gt])
            label.append(int(record["label"][gt] > 0))

    pred = np.array(pred)
    label = np.array(label)

    acc = (pred == label).sum() / len(pred)
    metric = torchmetrics.F1Score(task="binary", num_classes=2)
    f1 = float(metric(torch.from_numpy(pred), torch.from_numpy(label)))
    print(f"Accuracy {acc}")
    print(f"F1 {f1}")
    return acc, f1
