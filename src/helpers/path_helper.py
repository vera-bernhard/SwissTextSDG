import os
from typing import List
from pathlib import Path
from typing import List

DATASET_RESULT_PATH = os.path.join('data', 'results')
DATASET_RAW_PATH = os.path.join('data', 'raw')
DATASET_PROCESSED_PATH = os.path.join('data', 'processed')
MODEL_PATH = os.path.join('models')


def path_exists_or_create(path: str) -> bool:
    path = Path(path)
    path_exists = path.exists()

    if not path_exists:
        path.mkdir(exist_ok=True, parents=True)

    return path_exists


def file_exists_or_create(file_path: str) -> bool:
    file_path = Path(file_path)
    file_exists = os.path.isfile(file_path)

    if not file_exists:
        file_path.parent.mkdir(exist_ok=True, parents=True)

    return file_exists

# /data/processed
def dataset_processed_folder_path(dataset_name: str, seed: int = None) -> str:
    if seed:
        return os.path.join(DATASET_PROCESSED_PATH, dataset_name, f"seed_{seed}")
    return os.path.join(DATASET_PROCESSED_PATH, dataset_name)

def dataset_processed_file_path(dataset_name: str, file_name: str, seed: int = None) -> str:
    return os.path.join(dataset_processed_folder_path(dataset_name, seed=seed), file_name)

# /data/raw
def dataset_raw_file_path(file_name: str) -> str:
    return os.path.join(DATASET_RAW_PATH, file_name)

# /data/results
def dataset_results_folder_path() -> str:
    return DATASET_RESULT_PATH

# /data/results/subfolderA/subfolderB/...

def dataset_results_folder_path__with_subfolders(subfolder_list: List[str]) -> str:
    return os.path.join(dataset_results_folder_path(), *subfolder_list)

# /data/results/subfolderA/subfolderB/.../file.ext
def dataset_results_file_path__with_subfolders(subfolder_list: List[str], file_name: str) -> str:
    return os.path.join(dataset_results_folder_path(), *subfolder_list, file_name)


# /models
def experiment_file_path(experiment_name: str, file_name: str) -> str:
    return os.path.join(MODEL_PATH, experiment_name, file_name)


def experiment_config_path(experiment_name: str) -> str:
    return experiment_file_path(experiment_name, 'config.cfg')
