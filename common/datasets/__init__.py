from .wiki_train_dataset import WikiTrainSet
from .wiki_test_dataset import WikiTestSet
from functools import partial

dataset = {
    "train": partial(WikiTrainSet, data_root="/data/home/liyh/shulin/textf/kaggle_caption/common/dataset/data/train_set"),
    "test": partial(WikiTestSet, data_root="/data/home/liyh/shulin/textf/kaggle_caption/common/dataset/data/test_set"),
}
