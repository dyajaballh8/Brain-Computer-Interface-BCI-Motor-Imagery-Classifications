# Brain-Computer Interface (BCI) — Motor Imagery Classification

Classifying EEG motor-imagery trials (Left vs Right hand) end to end: exploration, preprocessing, feature extraction, and modeling.

## Pipeline

| Step | Notebook | Docs |
|---|---|---|
| 1. Exploratory Data Analysis | [01_EEG_EDA.ipynb](01_EEG_EDA.ipynb) | [README](README_01_EEG_EDA.md) |
| 2. Preprocessing | [02_EEG_Preprocessing.ipynb](02_EEG_Preprocessing.ipynb) | [README](README_02_EEG_Preprocessing.md) |
| 3. Modeling | [03_EEG_Modeling.ipynb](03_EEG_Modeling.ipynb) | [README](README_03_EEG_Modeling.md) |
| 4. Feature Extraction | [04_Feature_Extraction.ipynb](04_Feature_Extraction.ipynb) | — |
| Full pipeline | [main_pipeline.ipynb](main_pipeline.ipynb) | — |

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```
