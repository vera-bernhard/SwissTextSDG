from enum import Enum, IntEnum

from tqdm.auto import tqdm


# Enum to keep track and choose splitting methods for the
# train/test split in the dataset

class SplitMethod(Enum):
    RANDOM = 0
    PRE_SPLIT = 1
