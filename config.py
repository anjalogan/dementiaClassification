from pathlib import Path
import torch


def count_subdirectories(path: Path) -> int:
    subdirectories = [f for f in path.iterdir() if f.is_dir()]
    return len(subdirectories)


class Config:
    DATA_DIR = Path('data')
    TRAIN_DIR = DATA_DIR.joinpath('train')
    TEST_DIR = DATA_DIR.joinpath('test')
    VALID_DIR = DATA_DIR.joinpath('validate')

    NUM_CLASSES = count_subdirectories(DATA_DIR)

    IMAGE_TYPE = '.jpg'
    BATCH_SIZE = 32
    MODEL_NAME = 'resnet18'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAINING_PARAMS = 'training_hyperparams/imagenet_vit_train_params'

    CHECKPOINT_DIR = 'checkpoints'

    DEVICE = 'cpu'
