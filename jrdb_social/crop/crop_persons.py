from collections import defaultdict
from multiprocessing import cpu_count
import os
import tensorneko as N
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

from ..data import train_seqs, valid_seqs, test_seqs
from ..misc import parse_multi_simin_dict_as_list, parse_simin_dict


def process_bbox(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    x = x1 + w // 2
    y = y1 + h // 2

    margin = max(w, h) // 2

    x1, x2 = x - margin, x + margin
    y1, y2 = y - margin, y + margin
    return x1, x2, y1, y2


def filter_bbox_content(input_image, bbox):
    # only the bbox content is kept
    x1, y1, w, h = bbox
    new_image = np.zeros_like(input_image, dtype=np.uint8)
    new_image[y1:(y1+h), x1:(x1+w)] = input_image[y1:(y1+h), x1:(x1+w)]
    return new_image


def crop_person_video(person_data, video_name, person_id, out_folder, data_root):
    # save as video
    video_writer = cv2.VideoWriter(str(out_folder / f"{person_id}.avi"), cv2.VideoWriter_fourcc(*'XVID'), 30, (256, 256))
    
    for frame_name, bbox in person_data["box"]:
        x1, x2, y1, y2 = process_bbox(bbox)
        if os.path.exists(frame_path := f"{data_root}/images/image_stitched/{video_name}/{frame_name:06}.jpg"):
            frame = cv2.imread(frame_path)
            frame = filter_bbox_content(frame, bbox)
            frame = N.preprocess.crop_with_padding(frame, x1, x2, y1, y2, 0)
            frame = cv2.resize(frame, (256, 256))
            video_writer.write(frame)

    video_writer.release()


def main(data_root: str, split: str):
    if split == "train":
        seqs = train_seqs
    elif split == "valid":
        seqs = valid_seqs
    elif split == "test":
        seqs = test_seqs
    else:
        raise ValueError(f"Invalid split: {split}")
    
    for video_name in seqs:
        all_persons = defaultdict(dict) # key: person label_ID
        # labels = N.io.read.json(f"labels_2d_stitched_new_version_train_attribute/{video_name}.json")["labels"]
        labels = N.io.read.json(os.path.join(data_root, "labels", "labels_2d_activity_social_stitched", f"{video_name}.json"))["labels"]
        all_frame_names = sorted(labels.keys())
        for frame_name in all_frame_names:
            for entry in labels[frame_name]:
                label_id = entry["label_id"].replace(":", "_")
                if all_persons[label_id] == {}:
                    # not initialized, add labels
                    if len(entry["demographics_info"]) > 0:
                        label_record = {
                            "age": list(parse_simin_dict(entry["demographics_info"][0]["age"])),
                            "gender": list(parse_simin_dict(entry["demographics_info"][0]["gender"])),
                            "race": list(parse_simin_dict(entry["demographics_info"][0]["race"])),
                            "action": [list(each for each in parse_multi_simin_dict_as_list(entry["action_label"]))]
                        }
                    else:
                        label_record = {
                            "age": None,
                            "gender": None,
                            "race": None,
                            "action": None
                        }
                    all_persons[label_id]["label"] = label_record
                    all_persons[label_id]["box"] = []
                all_persons[label_id]["box"].append([int(frame_name[:-4]), entry["box"]])
        
        out_folder = Path(f"{data_root}/cropped/persons/{video_name}")
        out_folder.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = []
            
            for i, person_id in enumerate(all_persons.keys()):
                futures.append(executor.submit(crop_person_video, all_persons[person_id], video_name, person_id, out_folder, data_root))

            for future in tqdm(futures):
                future.result()

        N.io.write.json(str(out_folder / "labels.json"), dict(all_persons), fast=False)
