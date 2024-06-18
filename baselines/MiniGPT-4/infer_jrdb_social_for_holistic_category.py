from infer_jrdb_social import *
from collections import defaultdict
from jrdb_social.data import calculate_acc_and_f1_for_holistic_category, main_locations


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
                text = f"Do you see {label} in the video? Your answer should be yes or no. Please think and generate only the word as the answer."
            elif prompt == "main_locations":
                text = f"Do you see {label} in the video? Your answer should be yes or no. Please think and generate only the word as the answer."
            elif prompt == "pose":
                text = f"Do you see any group located on {label}? Your answer should be yes or no. Please think and generate only the word as the answer."
            elif prompt == "salient":
                text = f"Do you see any group near the {label}? Your answer should be yes or no. Please think and generate only the word as the answer."
            elif prompt == "intention":
                text = f"Do you see any group are {label}? Your answer should be yes or no. Please think and generate only the word as the answer."
            elif prompt == "interaction":
                text = f"Do you see any pair of people are {label}? Your answer should be yes or no. Please think and generate only the word as the answer."
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

            if "yes" in llm_message:
                pred = 1
                answers[label] += pred
                raws[label].append(llm_message)
            elif "no" in llm_message:
                pred = 0
                answers[label] += pred
                raws[label].append(llm_message)
            else:
                pred = 0
                answers[label] += pred
                raws[label].append(llm_message)

    answers = {k: int((v / len(raws[k])) >= 0.5 if len(raws[k]) > 0 else 0) for k, v in answers.items()}
    return answers, raws



def transform_main_location(label):
    ## single str label to onehot dict
    new_label = {}
    for loc in main_locations:
        if label is not None and loc in label:
            new_label[loc] = 1
        else:
            new_label[loc] = 0
    return new_label


def predict_holistic(max_trial, split):
    dataset = JRDBSocial.holistics(data_root, split)

    records = defaultdict(dict)

    for i in tqdm(range(len(dataset))):
        if i == 1: break
        datum = zip(*dataset[i])
        for video_path, genders_label, ages_label, races_label, poses_label, salients_label, intentions_label, main_localtions_label, interactions_label in datum:
            frames = load_video_frames(video_path, max_trial)
            
            gender_answers, gender_pred_raw = infer_holistics(frames, max_trial, ["male", "female"], "person")
            records["gender"][video_path.replace(data_root, ".")] = {"raws": gender_pred_raw, "answers": gender_answers, "label": genders_label}

            age_answers, age_pred_raw = infer_holistics(frames, max_trial, ages, "person")
            records["age"][video_path.replace(data_root, ".")] = {"raws": age_pred_raw, "answers": age_answers, "label": ages_label}

            race_answers, race_pred_raw = infer_holistics(frames, max_trial, races, "person")
            records["race"][video_path.replace(data_root, ".")] = {"raws": race_pred_raw, "answers": race_answers, "label": races_label}

            main_locations_answers, main_locations_pred_raw = infer_holistics(frames, max_trial, main_locations, "main_locations")
            records["main_locations"][video_path.replace(data_root, ".")] = {"raws": main_locations_pred_raw, "answers": main_locations_answers, "label": transform_main_location(main_localtions_label)}

            pose_answers, pose_pred_raw = infer_holistics(frames, max_trial, poses, "pose")
            records["pose"][video_path.replace(data_root, ".")] = {"raws": pose_pred_raw, "answers": pose_answers, "label": poses_label}

            salient_answers, salient_pred_raw = infer_holistics(frames, max_trial, salients, "salient")
            records["salient"][video_path.replace(data_root, ".")] = {"raws": salient_pred_raw, "answers": salient_answers, "label": salients_label}

            intention_answers, intention_pred_raw = infer_holistics(frames, max_trial, intentions, "intention")
            records["intention"][video_path.replace(data_root, ".")] = {"raws": intention_pred_raw, "answers": intention_answers, "label": intentions_label}

            interaction_answers, interaction_pred_raw = infer_holistics(frames, max_trial, interactions, "interaction")
            records["interaction"][video_path.replace(data_root, ".")] = {"raws": interaction_pred_raw, "answers": interaction_answers, "label": interactions_label}


    with open(f"records_minigpt4_holistic_category_{max_trial}_{split}.json", "w") as f:
        json.dump(dict(records), f, indent=4)

    if split != "test":
        print(f"Holistic Results for split={split}, max_trial={max_trial}")
        print("Gender")
        calculate_acc_and_f1_for_holistic_category(records["gender"])
        print("Age")
        calculate_acc_and_f1_for_holistic_category(records["age"])
        print("Race")
        calculate_acc_and_f1_for_holistic_category(records["race"])
        print("Main Location")
        calculate_acc_and_f1_for_holistic_category(records["main_locations"])
        print("Intention")
        calculate_acc_and_f1_for_holistic_category(records["intention"])
        print("Pose")
        calculate_acc_and_f1_for_holistic_category(records["pose"])
        print("Salient")
        calculate_acc_and_f1_for_holistic_category(records["salient"])
        print("Interaction")
        calculate_acc_and_f1_for_holistic_category(records["interaction"])


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

    # split = args.split
    # max_trial = args.max_trials
    # with open(f"records_minigpt4_holistic_category_{max_trial}_{split}.json", "r") as f:
    #     records = json.load(f)

    # if split != "test":
    #     print(f"Holistic Results for split={split}, max_trial={max_trial}")
    #     print("Gender")
    #     calculate_acc_and_f1_for_holistic_category(records["gender"])
    #     print("Age")
    #     calculate_acc_and_f1_for_holistic_category(records["age"])
    #     print("Race")
    #     calculate_acc_and_f1_for_holistic_category(records["race"])
    #     print("Main Localtion")
    #     calculate_acc_and_f1_for_holistic_category(records["main_locations"])
    #     print("Intention")
    #     calculate_acc_and_f1_for_holistic_category(records["intention"])
    #     print("Pose")
    #     calculate_acc_and_f1_for_holistic_category(records["pose"])
    #     print("Salient")
    #     calculate_acc_and_f1_for_holistic_category(records["salient"])
    #     print("Interaction")
    #     calculate_acc_and_f1_for_holistic_category(records["interaction"])