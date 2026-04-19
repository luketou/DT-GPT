import numpy
import random
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.local_paths import get_mimic_constants_path




if __name__ == '__main__':
    # Set random seed
    random.seed(0)
    numpy.random.seed(0)

    train_val_test_split = [0.8, 0.1, 0.1]

    constants_path = get_mimic_constants_path()

    # Load constants
    constants = pd.read_csv(constants_path)


    # Split data
    n = len(constants)
    indices = list(range(n))
    random.shuffle(indices)

    train_indices = indices[:int(n * train_val_test_split[0])]
    val_indices = indices[int(n * train_val_test_split[0]):int(n * (train_val_test_split[0] + train_val_test_split[1]))]
    test_indices = indices[int(n * (train_val_test_split[0] + train_val_test_split[1])):]

    # assert that they do not overlap
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0

    # assert that they make the full set
    assert len(set(train_indices).union(set(val_indices)).union(set(test_indices))) == n

    # Save splits
    constants.loc[train_indices, "dataset_split"] = "TRAIN"
    constants.loc[val_indices, "dataset_split"] = "VALIDATION"
    constants.loc[test_indices, "dataset_split"] = "TEST"

    # Save constants
    constants.to_csv(constants_path, index=False)

    

