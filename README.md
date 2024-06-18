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
