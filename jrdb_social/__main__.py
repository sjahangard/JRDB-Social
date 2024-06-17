import argparse
from .crop import crop_holistics, crop_persons, crop_interaction, crop_groups



def crop(args: argparse.Namespace):
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    for split in splits:
        if args.level == "person":
            crop_persons.main(args.data_root, split)
        elif args.level == "interaction":
            crop_interaction.main(args.data_root, split)
        elif args.level == "group":
            crop_groups.main(args.data_root, split)
        elif args.level == "holistic":
            crop_holistics.main(args.data_root, split)
        elif args.level == "all":
            crop_persons.main(args.data_root, split)
            crop_interaction.main(args.data_root, split)
            crop_groups.main(args.data_root, split)
            crop_holistics.main(args.data_root, split)
        else:
            raise ValueError(f"Unknown level: {args.level}")

def evaluate(args: argparse.Namespace):
    raise NotImplementedError("Not implemented yet")


def main():
    args = argparse.ArgumentParser()
    sub_parsers = args.add_subparsers()
    parser_crop = sub_parsers.add_parser("crop")
    parser_crop.set_defaults(func=crop)
    parser_crop.add_argument("level", type=str, choices=["person", "interaction", "group", "holistic", "all"])
    parser_crop.add_argument("--data_root", type=str, required=True)
    parser_crop.add_argument("--split", type=str, choices=["train", "val", "test", "all"], required=True)

    parser_evaluate = sub_parsers.add_parser("evaluate")
    parser_evaluate.set_defaults(func=evaluate)
    parser_evaluate.add_argument("level", type=str, choices=["person", "interaction", "group", "holistic", "all"])
    parser_evaluate.add_argument("--data_root", type=str, required=True)
    parser_evaluate.add_argument("--split", type=str, choices=["train", "val", "test", "all"], required=True)

    args = args.parse_args()
    
    args.func(args)


if __name__ == "__main__":
    main()
