from collections import defaultdict
import os
import numpy as np
import pandas as pd
import tensorneko_util as N
import cv2
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

from ..data import test_seqs, train_seqs, valid_seqs


def crop_holistic_video(data_root, video_name):
    # save as video
    video_writers = []
    for i in range(5):
        video_writers.append(cv2.VideoWriter(f"{data_root}/cropped/holistics/{video_name}/{i}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (512, 512)))

    for frame_name in sorted(os.listdir(frame_dir := f"{data_root}/images/image_stitched/{video_name}")):
        frame = cv2.imread(f"{frame_dir}/{frame_name}")  # (480, 3760, 3)

        for i in range(5):
            view = frame[:, i * 752:(i + 1) * 752, :]  # (480, 752, 3)
            # pad to 752 * 752
            view = N.preprocess.crop_with_padding(view, 0, 752, -136, 616, 0)  # (752, 752, 3)
            # resize to 512 * 512
            view = cv2.resize(view, (512, 512))
            video_writers[i].write(view)

    for video_writer in video_writers:
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

    for video_name in tqdm(seqs):
        (Path(data_root) / "cropped" / "holistics" / video_name).mkdir(parents=True, exist_ok=True)
        crop_holistic_video(data_root, video_name)
