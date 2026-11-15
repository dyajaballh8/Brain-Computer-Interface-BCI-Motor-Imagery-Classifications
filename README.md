# EEG LEFT/RIGHT - EDA and Preprocessing

## Folder structure
```
EEG_Task/
  01_EEG_EDA.ipynb             exploratory data analysis (8 cells)
  02_EEG_Preprocessing.ipynb   preprocessing pipeline (7 cells)
  labels.csv                   one label (Left / Right) per trial
  requirements.txt
  data/eeg_files/              <- put the Excel trial files here
  processed_data/              <- outputs of notebook 02
```

## How to run
1. `pip install -r requirements.txt`
2. Copy the Excel files into `data/eeg_files/`.
3. `jupyter notebook`, then run `01_EEG_EDA.ipynb` and afterwards `02_EEG_Preprocessing.ipynb` (Run All).
4. In Cell 1 of each notebook set the real sampling rate `FS` (default 256).

## Notes
- labels.csv has only a `label` column, so row i is matched with the i-th Excel file in natural sorted order.
  If your file order is different, add a `filename` column to labels.csv and it will be used instead.
- The number of Excel files must equal the number of rows in labels.csv.
- Outputs: `trial_label_mapping.csv` (EDA) and `processed_data/` (X_eeg.npy, y_eeg.npy, channels.json,
  config.json, kept_trials.csv, excluded_trials.csv).
