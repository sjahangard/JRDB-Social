# JRDB Social SDK

## Installation

```sh
pip install git+https://github.com/sjahangard/JRDB-Social.git
```

## Data Preprocessing 

```sh
jrdb_social crop <LEVEL> --data_root <DATA_ROOT> --split <SPLIT>
```

- `LEVEL`: `person`, `group`, `interaction`, `holistic`, `all`
- `SPLIT`: `train`, `valid`, `test`

Then the processed data and labels will be saved in `DATA_ROOT/cropped/<LEVEL>`

## Data Loader

```python
from jrdb_social.data import JRDBSocial

if __name__ == '__main__':
    # load persons data from JRDB dataset
    train_data = JRDBSocial.persons("../JRDB/train_dataset_with_activity", "train")
    
    for each in train_data:
        print(each)

    # load groups data from JRDB dataset
    train_data = JRDBSocial.groups("../JRDB/train_dataset_with_activity", "train")

    for each in train_data:
        print(each)

    # load interaction data from JRDB dataset
    train_data = JRDBSocial.interactions("../JRDB/train_dataset_with_activity", "train")

    for each in train_data:
        print(each)

    # load holistics data from JRDB dataset
    train_data = JRDBSocial.holistics("../JRDB/train_dataset_with_activity", "train")

    for each in train_data:
        print(each)
```

## Baselines

### MiniGPT-4

#### Getting Started

```bash
conda create -n minigpt4 python=3.10
conda activate minigpt4
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install huggingface-hub==0.18.0 matplotlib==3.7.0 psutil==5.9.4 iopath pyyaml regex tokenizers==0.13.2 tqdm transformers==4.30.0 timm==0.6.13 webdataset==0.2.48 omegaconf==2.3.0 opencv-python==4.7.0.72 decord==0.6.0 peft==0.2.0 sentence-transformers gradio==3.47.1 accelerate==0.20.3 bitsandbytes wandb
pip install git+https://github.com/sjahangard/JRDB-Social.git
```

Then you need to download `minigptv2_checkpoint.pth` from the [original repository](https://github.com/Vision-CAIR/MiniGPT-4) and put it in the `baselines/MiniGPT-4` directory.

#### Evaluation

Evaluate for `person`, `group`, `interaction` levels.
```bash
python infer_jrdb_social.py --data_root <DATA_ROOT> --type <LEVEL> --max_trials <ENSEMBLE> --split <SPLIT> --llama_path <PATH/TO/llama-2-7b-chat-hf>
```

Evaluate for `holistic` level.
```bash
# counting
python infer_jrdb_social_for_holistic.py --data_root <DATA_ROOT> --max_trials <ENSEMBLE> --split <SPLIT> --llama_path <PATH/TO/llama-2-7b-chat-hf>
# binary
python infer_jrdb_social_for_holistic_category.py --data_root <DATA_ROOT> --max_trials <ENSEMBLE> --split <SPLIT> --llama_path <PATH/TO/llama-2-7b-chat-hf>
```

## Reference

```bibtex
@inproceedings{jahangard2024jrdb,
  title={JRDB-Social: A Multifaceted Robotic Dataset for Understanding of Context and Dynamics of Human Interactions Within Social Groups},
  author={Jahangard, Simindokht and Cai, Zhixi and Wen, Shiki and Rezatofighi, Hamid},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22087--22097},
  year={2024}
}
```
