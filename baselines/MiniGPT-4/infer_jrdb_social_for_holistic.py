from infer_jrdb_social import *
from collections import defaultdict
from jrdb_social.data import calculate_acc_and_f1_for_holistic
import re


def infer_holistics(frames, num_trials, categories, prompt):
    assert len(frames) == num_trials

    raws = dict(zip(categories, [[] for _ in range(len(categories))]))
    answers = dict(zip(categories, [0] * len(categories)))

    for label in answers:
        for i in range(num_trials):
            img_list = []
            chat_state = CONV_VISION.copy()
            chat.upload_img(Image.fromarray(frames[i]), chat_state, img_list)

            if prompt == "person":
                text = f"How many {label} in the video? Your answer should be number. Please think and generate only the number as the answer."
            elif prompt == "pose":
                text = f"How many groups of people located on {label}? Your answer should be number. Please think and generate only the number as the answer."
            elif prompt == "salient":
                text = f"How many groups of people near the {label}? Your answer should be number. Please think and generate only the number as the answer."
            elif prompt == "intention":
                text = f"How many groups of people are {label}? Your answer should be number. Please think and generate only the number as the answer."
            elif prompt == "interaction":
                text = f"How many pairs of people are {label}? Your answer should be number. Please think and generate only the number as the answer."
            else:
                raise ValueError(prompt)
                
            chat.ask("[vqa] " + text, chat_state)
            chat.encode_img(img_list)
            llm_message = chat.answer(conv=chat_state,
                                      img_list=img_list,
                                      temperature=0.6,
                                      max_new_tokens=500,
                                      max_length=2000)[0]
            llm_message = llm_message.lower().strip(".")

            try:
                pred = int(llm_message)
                answers[label] += pred
                raws[label].append(llm_message)
            except:
                try:
                    pred = int(re.search(r"\d+", llm_message).group(0))
                    answers[label] += pred
                    raws[label].append(llm_message)
                except:
                    pass

    answers = {k: int(v / len(raws[k]) if len(raws[k]) > 0 else 0) for k, v in answers.items()}
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
    return answers, raws


def predict_holistic(max_trial, split):
    dataset = JRDBSocial.holistics(data_root, split)

    records = defaultdict(dict)

    for i in tqdm(range(len(dataset))):
        datum = zip(*dataset[i])
        for video_path, genders_label, ages_label, races_label, poses_label, salients_label, intentions_label, main_localtions_label, interactions_label in datum:
            frames = load_video_frames(video_path, max_trial)
            
            gender_answers, gender_pred_raw = infer_holistics(frames, max_trial, ["male", "female"], "person")
            records["gender"][video_path.replace(data_root, ".")] = {"raws": gender_pred_raw, "answers": gender_answers, "label": genders_label}

            age_answers, age_pred_raw = infer_holistics(frames, max_trial, ages, "person")
            records["age"][video_path.replace(data_root, ".")] = {"raws": age_pred_raw, "answers": age_answers, "label": ages_label}

            race_answers, race_pred_raw = infer_holistics(frames, max_trial, races, "person")
            records["race"][video_path.replace(data_root, ".")] = {"raws": race_pred_raw, "answers": race_answers, "label": races_label}

            pose_answers, pose_pred_raw = infer_holistics(frames, max_trial, poses, "pose")
            records["pose"][video_path.replace(data_root, ".")] = {"raws": pose_pred_raw, "answers": pose_answers, "label": poses_label}

            salient_answers, salient_pred_raw = infer_holistics(frames, max_trial, salients, "salient")
            records["salient"][video_path.replace(data_root, ".")] = {"raws": salient_pred_raw, "answers": salient_answers, "label": salients_label}

            intention_answers, intention_pred_raw = infer_holistics(frames, max_trial, intentions, "intention")
            records["intention"][video_path.replace(data_root, ".")] = {"raws": intention_pred_raw, "answers": intention_answers, "label": intentions_label}

            main_localtion_answers, main_localtion_pred_raw = infer_main_location(frames, max_trial)
            records["main_location"][video_path.replace(data_root, ".")] = {"raws": main_localtion_pred_raw, "answers": main_localtion_answers, "label": main_localtions_label}

            interaction_answers, interaction_pred_raw = infer_holistics(frames, max_trial, interactions, "interaction")
            records["interaction"][video_path.replace(data_root, ".")] = {"raws": interaction_pred_raw, "answers": interaction_answers, "label": interactions_label}


    with open(f"records_minigpt4_holistic_{max_trial}_{split}.json", "w") as f:
        json.dump(dict(records), f, indent=4)

    if split != "test":
        print(f"Holistic Results for split={split}, max_trial={max_trial}")
        print("Gender")
        calculate_acc_and_f1_for_holistic(records["gender"])
        print("Age")
        calculate_acc_and_f1_for_holistic(records["age"])
        print("Race")
        calculate_acc_and_f1_for_holistic(records["race"])
        print("Main Location")
        main_location_preds = [(max(v["answers"], key=lambda x: v["answers"][x]) if max(v["answers"].values()) > 0 else None) for k, v in records["main_location"].items()]
        main_location_labels = [v["label"] for k,v in records["main_location"].items()]
        main_location_preds = [main_location_preds[i] for i in range(len(main_location_labels)) if main_location_labels[i] is not None]
        main_location_labels = [each for each in main_location_labels if each is not None]
        calculate_acc_and_f1(main_location_preds, main_location_labels, main_locations)
        print("Intention")
        calculate_acc_and_f1_for_holistic(records["intention"])
        print("Pose")
        calculate_acc_and_f1_for_holistic(records["pose"])
        print("Salient")
        calculate_acc_and_f1_for_holistic(records["salient"])
        print("Interaction")
        calculate_acc_and_f1_for_holistic(records["interaction"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_trials", type=int, required=True)
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], required=True)
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
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)

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

    predict_holistic(args.max_trials, args.split)
