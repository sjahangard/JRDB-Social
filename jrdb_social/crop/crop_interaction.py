from collections import defaultdict
import os
import numpy as np
import pandas as pd
import tensorneko as N
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple


from ..data import train_seqs, valid_seqs, test_seqs


@dataclass
class InteractionEntry:
    video_name: str
    group_id: int
    persons: Tuple[int, ...]
    start: int
    end: Optional[int]
    interaction: str
    difficulty: int

    entries = dict()  # key: (video_name, group_id, person_1, person_2, start, end, interaction, difficulty), value: List[InteractionEntry]
    prev_frame_entries = set()
    current_frame_entries = set()

    def __hash__(self) -> int:
        return hash((self.video_name, self.group_id, self.persons, self.start, self.end, self.interaction, self.difficulty))

    @classmethod
    def add(cls, video_name: str, group_id: int, persons_in: Tuple[int, int], current_timestamp: int, interaction: str, difficulty: int):
        persons = tuple(sorted(persons_in))
        for entry in cls.prev_frame_entries:
            if entry.video_name == video_name and entry.group_id == group_id \
                and entry.persons[0] == persons[0] and entry.persons[1] == persons[1] \
                and entry.difficulty == difficulty and entry.interaction == interaction:

                entry.end = current_timestamp
                cls.current_frame_entries.add(entry)
                return
        else:
            cls.current_frame_entries.add(cls(video_name, group_id, persons, current_timestamp, current_timestamp, interaction, difficulty))

    @classmethod
    def flush(cls):
        # find these entries which is in the previous frame but not in the current frame
        for entry in cls.prev_frame_entries:
            if entry not in cls.current_frame_entries:
                cls.entries[(entry.video_name, entry.group_id, tuple(entry.persons), entry.start, entry.end, entry.interaction, entry.difficulty)] = entry
        cls.prev_frame_entries = cls.current_frame_entries
        cls.current_frame_entries = set()

    @classmethod
    def clear(cls):
        cls.prev_frame_entries = set()
        cls.current_frame_entries = set()
        cls.entries = dict()


def process_bbox(bboxes):
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


def crop_interaction_video(interaction_data, video_name, file_name, out_folder, data_root):
    # save as video
    video_writer = cv2.VideoWriter(
        str(out_folder / file_name), cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512))

    frame_names = sorted(interaction_data["box"].keys())

    for frame_name in frame_names:
        bboxes = interaction_data["box"][frame_name]

        x1, x2, y1, y2 = process_bbox(bboxes)
        
        frame = cv2.imread(
            f"{data_root}/images/image_stitched/{video_name}/{frame_name:06}.jpg")
        frame = N.preprocess.crop_with_padding(
            frame, int(x1), int(x2), int(y1), int(y2), 0)
        frame = cv2.resize(frame, (512, 512))
        video_writer.write(frame)

    video_writer.release()


def main(data_root, split):
    if split == "train":
        seqs = train_seqs
    elif split == "valid":
        seqs = valid_seqs
    elif split == "test":
        seqs = test_seqs
    else:
        raise ValueError(f"Invalid split: {split}")

    for video_name in seqs:
        # key: cluster_ID, value: frame_names
        all_interactions = defaultdict(dict)
        bbox_labels = N.io.read.json(os.path.join(data_root, "labels", "labels_2d_activity_social_stitched", f"{video_name}.json"))["labels"]
        # bbox_labels = N.io.read.json(f"labels_2d_stitched/{video_name}.json")["labels"]

        # build labels
        for key, value in bbox_labels.items():
            current_timestamp = int(key[:-4])
            for each in value:
                if len(each["H-interaction"]) == 0:
                    continue

                person_id_1 = int(each["label_id"].replace("pedestrian:", ""))
                for interaction in each["H-interaction"]:
                    person_id_2 = int(interaction["pair"].replace("pedestrian:", ""))
                    for inter_label, difficulty in interaction["inter_labels"].items():
                        InteractionEntry.add(
                            video_name, each["social_group"]["cluster_ID"], [person_id_1, person_id_2], current_timestamp, inter_label, difficulty
                        )
            InteractionEntry.flush()
        InteractionEntry.flush()

        for row in InteractionEntry.entries.values():
            file_name = f"{row.group_id}_{row.start}_{row.end}_{'_'.join(map(str, row.persons))}.avi"

            if file_name in all_interactions:
                all_interactions[file_name]["interaction"].add(row.interaction)
                continue

            all_interactions[file_name]["interaction"] = {row.interaction}
            all_interactions[file_name]["box"] = dict()

            for frame_name in range(row.start, row.end + 1):
                frame_data = pd.DataFrame(bbox_labels[f"{frame_name:06}.jpg"])

                # filter the gruop
                frame_data = frame_data[frame_data.social_group.apply(
                    lambda x: x["cluster_ID"] == row.group_id)]

                # filter the persons
                frame_data = frame_data[frame_data.label_id.apply(
                    lambda x: int(x.replace("pedestrian:", "")) in row.persons)]

                # record the boxes
                try:
                    all_interactions[file_name]["box"][frame_name] = frame_data.box.tolist()
                except:
                    print(video_name, frame_name,
                          row.group_id, file_name, row.persons)
                    raise ValueError

                if len(all_interactions[file_name]["box"][frame_name]) == 0:
                    del all_interactions[file_name]["box"][frame_name]

        out_folder = Path(f"{data_root}/cropped/interactions/{video_name}")
        out_folder.mkdir(parents=True, exist_ok=True)

        # for file_name, interaction_data in all_interactions.items():
        #     crop_interaction_video(interaction_data, video_name, file_name, out_folder, data_root)

        with ProcessPoolExecutor(8) as executor:
            futures = []

            for file_name, interaction_data in all_interactions.items():
                futures.append(executor.submit(
                    crop_interaction_video, interaction_data, video_name, file_name, out_folder, data_root))

            for future in tqdm(futures):
                future.result()

        for key, value in all_interactions.items():
            all_interactions[key]["interaction"] = list(value["interaction"])

        N.io.write.json(f"{data_root}/cropped/interactions/{video_name}/labels.json", dict(all_interactions), fast=False)

        InteractionEntry.clear()

if __name__ == "__main__":
    main()
