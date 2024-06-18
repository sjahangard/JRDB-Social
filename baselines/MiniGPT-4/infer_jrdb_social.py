import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import tensorneko_util as N
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
import torchmetrics

from jrdb_social.data import JRDBSocial, train_seqs, valid_seqs, test_seqs, ages, races, main_locations, intentions, interactions, poses, salients


def extract_substrings(string):
    # first check if there is no-finished bracket
    index = string.rfind('}')
    if index != -1:
        string = string[:index + 1]

    pattern = r'<p>(.*?)\}(?!<)'
    matches = re.findall(pattern, string)
    substrings = [match for match in matches]

    return substrings


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "/tmp/gradio" + file_name
    visual_img.save(file_path)
    return file_path


def mask2bbox(mask):
    if mask is None:
        return ''
    mask = mask.resize([100, 100], resample=Image.NEAREST)
    mask = np.array(mask)[:, :, 0]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.sum():
        # Get the top, bottom, left, and right boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = '{{<{}><{}><{}><{}>}}'.format(cmin, rmin, cmax, rmax)
    else:
        bbox = ''

    return bbox


def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']

    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)

    return text


def reverse_escape(text):
    md_chars = ['\\<', '\\>']

    for char in md_chars:
        text = text.replace(char, char[1:])

    return text


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (210, 210, 0),
    (255, 0, 255),
    (0, 255, 255),
    (114, 128, 250),
    (0, 165, 255),
    (0, 128, 0),
    (144, 238, 144),
    (238, 238, 175),
    (255, 191, 0),
    (0, 128, 0),
    (226, 43, 138),
    (255, 0, 255),
    (0, 215, 255),
]

color_map = {
    f"{color_id}": f"#{hex(color[2])[2:].zfill(2)}{hex(color[1])[2:].zfill(2)}{hex(color[0])[2:].zfill(2)}" for
    color_id, color in enumerate(colors)
}

used_colors = colors


def visualize_all_bbox_together(image, generation):
    if image is None:
        return None, ''

    generation = html.unescape(generation)

    image_width, image_height = image.size
    image = image.resize([500, int(500 / image_width * image_height)])
    image_width, image_height = image.size

    string_list = extract_substrings(generation)
    if string_list:  # it is grounding or detection
        mode = 'all'
        entities = defaultdict(list)
        i = 0
        j = 0
        for string in string_list:
            try:
                obj, string = string.split('</p>')
            except ValueError:
                print('wrong string: ', string)
                continue
            bbox_list = string.split('<delim>')
            flag = False
            for bbox_string in bbox_list:
                integers = re.findall(r'-?\d+', bbox_string)
                if len(integers) == 4:
                    x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
                    left = x0 / bounding_box_size * image_width
                    bottom = y0 / bounding_box_size * image_height
                    right = x1 / bounding_box_size * image_width
                    top = y1 / bounding_box_size * image_height

                    entities[obj].append([left, bottom, right, top])

                    j += 1
                    flag = True
            if flag:
                i += 1
    else:
        integers = re.findall(r'-?\d+', generation)

        if len(integers) == 4:  # it is refer
            mode = 'single'

            entities = list()
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            left = x0 / bounding_box_size * image_width
            bottom = y0 / bounding_box_size * image_height
            right = x1 / bounding_box_size * image_width
            top = y1 / bounding_box_size * image_height
            entities.append([left, bottom, right, top])
        else:
            # don't detect any valid bbox to visualize
            return None, ''

    if len(entities) == 0:
        return None, ''

    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)

    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):

        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    indices = list(range(len(entities)))

    new_image = image.copy()

    previous_bboxes = []
    # size of text
    text_size = 0.5
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 2
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 2

    # num_bboxes = sum(len(x[-1]) for x in entities)
    used_colors = colors  # random.sample(colors, k=num_bboxes)

    color_id = -1
    for entity_idx, entity_name in enumerate(entities):
        if mode == 'single' or mode == 'identify':
            bboxes = entity_name
            bboxes = [bboxes]
        else:
            bboxes = entities[entity_name]
        color_id += 1
        for bbox_id, (x1_norm, y1_norm, x2_norm, y2_norm) in enumerate(bboxes):
            skip_flag = False
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm), int(y1_norm), int(x2_norm), int(y2_norm)

            color = used_colors[entity_idx % len(used_colors)]  # tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            if mode == 'all':
                l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

                x1 = orig_x1 - l_o
                y1 = orig_y1 - l_o

                if y1 < text_height + text_offset_original + 2 * text_spaces:
                    y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                    x1 = orig_x1 + r_o

                # add text background
                (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size,
                                                               text_line)
                text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (
                            text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

                for prev_bbox in previous_bboxes:
                    if computeIoU((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']) > 0.95 and \
                            prev_bbox['phrase'] == entity_name:
                        skip_flag = True
                        break
                    while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox['bbox']):
                        text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                        text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                        y1 += (text_height + text_offset_original + 2 * text_spaces)

                        if text_bg_y2 >= image_h:
                            text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                            text_bg_y2 = image_h
                            y1 = image_h
                            break
                if not skip_flag:
                    alpha = 0.5
                    for i in range(text_bg_y1, text_bg_y2):
                        for j in range(text_bg_x1, text_bg_x2):
                            if i < image_h and j < image_w:
                                if j < text_bg_x1 + 1.35 * c_width:
                                    # original color
                                    bg_color = color
                                else:
                                    # white
                                    bg_color = [255, 255, 255]
                                new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(
                                    np.uint8)

                    cv2.putText(
                        new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
                    )

                    previous_bboxes.append(
                        {'bbox': (text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), 'phrase': entity_name})

    if mode == 'all':
        def color_iterator(colors):
            while True:
                for color in colors:
                    yield color

        color_gen = color_iterator(colors)

        # Add colors to phrases and remove <p></p>
        def colored_phrases(match):
            phrase = match.group(1)
            color = next(color_gen)
            return f'<span style="color:rgb{color}">{phrase}</span>'

        generation = re.sub(r'{<\d+><\d+><\d+><\d+>}|<delim>', '', generation)
        generation_colored = re.sub(r'<p>(.*?)</p>', colored_phrases, generation)
    else:
        generation_colored = ''

    pil_image = Image.fromarray(new_image)
    return pil_image, generation_colored


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Upload your image and chat',
                                                                    interactive=True), chat_state, img_list


def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag


def example_trigger(text_input, image, upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list or replace_flag == 1:
        replace_flag = 1

    return upload_flag, replace_flag


def load_video_frames(video_path, samples):
    video = N.io.read.video(video_path).video
    # get several samples, make sure it's centered.
    n_frames = video.shape[0]
    indexes = (n_frames / (samples + 1) * np.arange(1, samples + 1)).astype(int)
    return video[indexes]


def infer_gender(frames, num_trials):
    assert len(frames) == num_trials

    raws = []
    answers = {"male": 0, "female": 0}

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask("[vqa] What is the gender of the person in the center of the video? Your answer should be one of male or female. Please think and generate only one word as the answer.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=0.6,
                                  max_new_tokens=500,
                                  max_length=2000)[0]
        llm_message = llm_message.lower().strip(".")
        raws.append(llm_message)

        # match single word output
        if llm_message in ("male", "men", "man"):
            answers["male"] += 1
            continue
        elif llm_message in ("female", "women", "woman"):
            answers["female"] += 1
            continue

        # if not match, match multiple words
        if " man" in llm_message or " male" in llm_message or " men" in llm_message:
            answers["male"] += 1

        if " woman" in llm_message or " female" in llm_message or " women" in llm_message:
            answers["female"] += 1

    return answers, raws


def infer_age(frames, num_trials):
    assert len(frames) == num_trials

    raws = []
    answers = dict(zip(ages, [0] * len(ages)))

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask("[vqa] What is the age of the person in the center of the video? Your answer should be one of young_adulthood, middle_adulthood, late_adulthood, adolescence or childhood. Please think and generate only one word as the answer.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=0.6,
                                  max_new_tokens=500,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)
        if "young_adulthood" in llm_message or "young adulthood" in llm_message or "young" in llm_message:
            answers["young_adulthood"] += 1

        if "middle_adulthood" in llm_message or "middle adulthood" in llm_message or "middle" in llm_message:
            answers["middle_adulthood"] += 1

        if "late_adulthood" in llm_message or "late adulthood" in llm_message or "old" in llm_message:
            answers["late_adulthood"] += 1

        if "adolescence" in llm_message or "adolescent" in llm_message or "teen" in llm_message or "teenager" in llm_message:
            answers["adolescence"] += 1

        if "childhood" in llm_message or "child" in llm_message or "kid" in llm_message:
            answers["childhood"] += 1

    return answers, raws


def infer_race(frames, num_trials):
    assert len(frames) == num_trials

    answers = dict(zip(races, [0] * len(races)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask("[vqa] What is the race of the person in the center of the video? Your answer should be one of caucasian, asians, black or others. Please think and generate only one word as the answer.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=0.6,
                                  max_new_tokens=500,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)
        if "caucasian" in llm_message or "white" in llm_message:
            answers["caucasian"] += 1

        if "asians" in llm_message or " asian" in llm_message:  # add a space for asian to avoid matching with caucasian
            answers["asians"] += 1

        if "african_american" in llm_message or "black" in llm_message:
            answers["black"] += 1

        if "others" in llm_message or "other" in llm_message:
            answers["others"] += 1

    return answers, raws


def infer_main_location(frames, num_trials):
    assert len(frames) == num_trials
    answers = dict(zip(main_locations, [0] * len(main_locations)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask(
            f"[vqa] What is the main location of the groups of people in the video? Your answer should be one of the following: {', '.join(main_locations)}. Please think and generate only one word as the answer.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)

        for main_location in main_locations:
            if main_location in llm_message \
                or main_location.replace("_", " ") in llm_message \
                or main_location.replace("/", " ") in llm_message \
                or main_location.replace("_", " ").replace("/", " ") in llm_message:
                answers[main_location] += 1
            else:
                for each_main_location in main_location.split("/"):
                    if each_main_location in llm_message:
                        answers[main_location] += 1
                        break
    return answers, raws


def infer_intention(frames, num_trials):
    assert len(frames) == num_trials
    answers = dict(zip(intentions, [0] * len(intentions)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask(
            f"[vqa] What is the intention of the groups of people in the video? Your answer should be one or multiple of the following: {', '.join(intentions)}. Please think and list the possible intentions.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)

        for intention in intentions:
            if intention in llm_message \
                or intention.replace("_", " ") in llm_message \
                or intention.replace("/", " ") in llm_message \
                or intention.replace("_", " ").replace("/", " ") in llm_message:
                answers[intention] += 1
    return answers, raws


def infer_poses(frames, num_trials):
    assert len(frames) == num_trials
    answers = dict(zip(poses, [0] * len(poses)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)
        
        chat.ask(
            f"[vqa] Where is the place of the person in the video? Your answer should be one of the following: {', '.join(poses)}. Please think and generate only one word as the answer.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)

        for pose in poses:
            if pose in llm_message \
                or pose.replace("_", " ") in llm_message \
                or pose.replace("(", " ").replace(")", " ") in llm_message \
                or pose.replace("(", " ").replace(")", " ").replace("/", " ") in llm_message:
                answers[pose] += 1
    return answers, raws
    

def infer_salients(frames, num_trials):
    assert len(frames) == num_trials
    answers = dict(zip(salients, [0] * len(salients)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)
        
        chat.ask(
            f"[vqa] What are the objects of the person in the video nearby? Your answer should be one or multiple of the following: {', '.join(salients)}. Please think and list all possible answers.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)

        for salient in salients:
            if salient in llm_message \
                or salient.replace("_", " ") in llm_message \
                or salient.replace("(", " ").replace(")", " ") in llm_message \
                or salient.replace("(", " ").replace(")", " ").replace("/", " ") in llm_message:
                answers[salient] += 1
    return answers, raws
        

def infer_interaction(frames, num_trials):
    answers = dict(zip(interactions, [0] * len(interactions)))
    raws = []

    for i in range(num_trials):
        img_list = []
        chat_state = CONV_VISION.copy()
        chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

        chat.ask(
            f"[vqa] What is the interaction between the people in the video? Your answer should be one or multiple of the following: {', '.join(interactions)}. Please think and list all possible answers.", chat_state)
        chat.encode_img(img_list)
        llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=1,
                                  temperature=1.0,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
        llm_message = llm_message.lower()
        raws.append(llm_message)

        for interaction in interactions:
            for match_str in (interaction, interaction.replace(" together", "")):
                if match_str in llm_message \
                    or match_str.replace("_", " ") in llm_message \
                    or match_str.replace("(", " ").replace(")", " ") in llm_message \
                    or match_str.replace("(", " ").replace(")", " ").replace("/", " ") in llm_message:
                    answers[interaction] += 1
    return answers, raws


def calculate_acc_and_f1(preds, labels, categories):
    preds = torch.tensor([categories.index(each) if each is not None else -1 for each in preds])
    labels = torch.tensor([categories.index(each) for each in labels])

    # calculate accuracy
    total_items = len(labels)
    correct = (preds == labels).sum().item()

    acc = correct / total_items
    print(f"Accuracy {correct} / {total_items} = {acc}")

    # calculate f1
    if len(categories) == 2:
        metric = torchmetrics.F1Score(task="binary", num_classes=len(categories))
        # assign the None prediction as the wrong class for easier calculation of f1
        preds[preds == -1] = 1 - labels[preds == -1]
    else:
        metric = torchmetrics.F1Score(task="multiclass", num_classes=len(categories), average="macro")
        # assign the None prediction as the wrong class for easier calculation of f1
        preds[preds == -1] = (labels[preds == -1] + 1) % len(categories)

    f1 = metric(preds, labels)
    print(f"F1 {f1}")
    return correct, total_items, acc, float(f1)


def calculate_acc_and_f1_for_multilabel(preds, labels, categories):
    preds = np.stack([N.util.sparse2binary(np.array([categories.index(each1) for each1 in each], dtype=int), length=len(categories)) for each in preds])
    labels = np.stack([N.util.sparse2binary(np.array([categories.index(each1) for each1 in each], dtype=int), length=len(categories)) for each in labels])

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


def predict_persons(max_trial, split):

    persons_dataset = JRDBSocial.persons(data_root, split)

    gender_preds = []
    gender_labels = []
    gender_records = {}

    age_preds = []
    age_labels = []
    age_records = {}

    race_preds = []
    race_labels = []
    race_records = {}

    for i, (video_path, gender_label, age_label, race_label) in tqdm(enumerate(persons_dataset), total=len(persons_dataset)):
        assert os.path.exists(video_path)

        if gender_label is None and age_label is None and race_label is None:
            continue

        try:
            frames = load_video_frames(video_path, max_trial)
        except:
            print(video_path)
            continue

        if gender_label is not None:
            gender_labels.append(gender_label)
            gender_answers, gender_pred_raw = infer_gender(frames, max_trial)

            if sum(gender_answers.values()) != 0:
                gender_pred = max(gender_answers, key=lambda x: gender_answers[x])
            else:
                gender_pred = None
            gender_preds.append(gender_pred)
            gender_records[video_path.replace(data_root, ".")] = {
                "raws": gender_pred_raw,
                "answers": gender_answers,
                "pred": gender_pred
            }

        if age_label is not None:
            age_labels.append(age_label)
            age_answers, age_pred_raw = infer_age(frames, max_trial)

            if sum(age_answers.values()) != 0:
                age_pred = max(age_answers, key=lambda x: age_answers[x])
            else:
                age_pred = None
            age_preds.append(age_pred)
            age_records[video_path.replace(data_root, ".")] = {
                "raws": age_pred_raw,
                "answers": age_answers,
                "pred": age_pred
            }

        if race_label is not None:
            race_labels.append(race_label)
            race_answers, race_pred_raw = infer_race(frames, max_trial)

            if sum(race_answers.values()) != 0:
                race_pred = max(race_answers, key=lambda x: race_answers[x])
            else:
                race_pred = None
            race_preds.append(race_pred)
            race_records[video_path.replace(data_root, ".")] = {
                "raws": race_pred_raw,
                "answers": race_answers,
                "pred": race_pred
            }

    with open(f"preds_minigpt4_{args.type}_{args.max_trials}_{args.split}.json", "w") as f:
        json.dump({"gender": gender_records, "age": age_records, "race": race_records}, f, indent=4, ensure_ascii=False)

    if split != "test":
        print("Gender")
        calculate_acc_and_f1(gender_preds, gender_labels, ["male", "female"])
        print("Age")
        calculate_acc_and_f1(age_preds, age_labels, ages)
        print("Race")
        calculate_acc_and_f1(race_preds, race_labels, races)
    

def predict_groups(max_trial, split):
    groups_dataset = JRDBSocial.groups(data_root, split)

    main_location_preds = []
    main_location_labels = []
    main_location_records = {}

    intention_preds = []
    intention_labels = []
    intention_records = {}

    poses_preds = []
    poses_labels = []
    poses_records = {}

    salients_preds = []
    salients_labels = []
    salients_records = {}

    for i, (video_path, local_location_label, salient_label, intention_label, main_location_label) in enumerate(tqdm(groups_dataset)):
        frames = load_video_frames(video_path, max_trial)
        if main_location_label is not None:
            main_location_labels.append(main_location_label)
            main_location_answers, main_location_pred_raw = infer_main_location(frames, max_trial)
            if sum(main_location_answers.values()) != 0:
                main_location_pred = max(main_location_answers, key=lambda x: main_location_answers[x])
            else:
                main_location_pred = None
            main_location_preds.append(main_location_pred)
            main_location_records[video_path.replace(data_root, ".")] = {
                "raws": main_location_pred_raw,
                "answers": main_location_answers,
                "pred": main_location_pred
            }

        if intention_label is not None:
            intention_labels.append(intention_label)
            intention_answers, intention_pred_raw = infer_intention(frames, max_trial)
            
            intention_pred = [k for k, v in intention_answers.items() if v > max_trial / 2]
            intention_preds.append(intention_pred)
            intention_records[video_path.replace(data_root, ".")] = {
                "raws": intention_pred_raw,
                "answers": intention_answers,
                "pred": intention_pred
            }

        if local_location_label is not None:
            poses_labels.append(local_location_label)
            poses_answers, poses_pred_raw = infer_poses(frames, max_trial)
            if sum(poses_answers.values()) != 0:
                poses_pred = max(poses_answers, key=lambda x: poses_answers[x])
            else:
                poses_pred = None

            poses_preds.append(poses_pred)
            poses_records[video_path.replace(data_root, ".")] = {
                "raws": poses_pred_raw,
                "answers": poses_answers,
                "pred": poses_pred
            }

        if salient_label is not None:
            salients_labels.append(salient_label)
            salients_answers, salients_pred_raw = infer_salients(frames, max_trial)

            salients_pred = [k for k, v in salients_answers.items() if v > max_trial / 2]

            salients_preds.append(salients_pred)
            salients_records[video_path.replace(data_root, ".")] = {
                "raws": salients_pred_raw,
                "answers": salients_answers,
                "pred": salients_pred
            }

    with open(f"preds_minigpt4_{args.type}_{args.max_trials}_{args.split}.json", "w") as f:
        json.dump({"main_location": main_location_records, "intention": intention_records, "poses": poses_records, "salients": salients_records}, f, indent=4, ensure_ascii=False)
    
    if split != "test":
        print("Main Location")
        calculate_acc_and_f1(main_location_preds, main_location_labels, main_locations)
        print("Intention")
        calculate_acc_and_f1_for_multilabel(intention_preds, intention_labels, intentions)
        print("Pose Location")
        calculate_acc_and_f1(poses_preds, poses_labels, poses)
        print("Salient Location")
        calculate_acc_and_f1_for_multilabel(salients_preds, salients_labels, salients)



def predict_interactions(max_trial, split):

    interaction_datasets = JRDBSocial.interactions(data_root, split)
    interaction_records = {}
    interaction_preds = []
    interaction_labels = []

    for i, (video_path, interaction_label) in enumerate(tqdm(interaction_datasets)):
        try:
            frames = load_video_frames(video_path, max_trial)
        except ValueError:
            print("Failed to load video", video_path)
            continue
        interaction_labels.append(interaction_label)
        interaction_answers, interaction_pred_raw = infer_interaction(frames, max_trial)
        interaction_pred = [k for k, v in interaction_answers.items() if v > max_trial / 2]
        interaction_preds.append(interaction_pred)
        interaction_records[video_path.replace(data_root, ".")] = {
            "raws": interaction_pred_raw,
            "answers": interaction_answers,
            "pred": interaction_pred
        }
    
    with open(f"preds_minigpt4_{args.type}_{args.max_trials}_{args.split}.json", "w") as f:
        json.dump({"interactions": interaction_records}, f, indent=4, ensure_ascii=False)

    if split != "test":
        calculate_acc_and_f1_for_multilabel(interaction_preds, interaction_labels, interactions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo")
    # parser.add_argument("--model", choices=["7b", "13b"], required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--type", choices=["persons", "groups", "interactions"], required=True)
    parser.add_argument("--max_trials", type=int, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], required=True)
    parser.add_argument("--llama_path", type=str, required=True)
    args = parser.parse_args()

    data_root = args.data_root
    print('Initializing Chat')
    class Args:
        cfg_path = 'eval_configs/minigpt4v2_eval.yaml'
        gpu_id = 0
        options = None
    args.cfg_path = Args.cfg_path
    args.gpu_id = Args.gpu_id
    args.options = Args.options
    if args.split == "valid":
        seqs = valid_seqs
    elif args.split == "train":
        seqs = train_seqs
    elif args.split == "test":
        seqs = test_seqs

    cfg = Config(args)
    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config, args.llama_path).to(device)
    bounding_box_size = 100

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    model = model.eval()

    CONV_VISION = Conversation(
        system="",
        roles=(r"<s>[INST] ", r" [/INST]"),
        messages=[],
        offset=2,
        sep_style=SeparatorStyle.SINGLE,
        sep="",
    )

    chat = Chat(model, vis_processor, device=device)

    eval(f"predict_{args.type}({args.max_trials}, args.split)")
