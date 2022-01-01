# Preprocess for MASS-SS3


## Requirements

- Python 3.6
- numpy 1.15.4
- scipy 1.1.0

## Usage

- You can use the process_data.py to compute DE and PSD features of a set of data.

- Data preparation for MI

1. Get the dataset, and extract useful data to save as .mat file.

2. Generate 4 fold data set, using mixed subjects scheme.

    - Default feature storage path is /test/trial_feature.npz
    - 4 fold data set is default stored as /test/XXXX.npz
3. You can use generate_feature.py to generate the initial adjency matrix.
    ```shell
    python process_data.py
    python generate_feature.py
    ```


