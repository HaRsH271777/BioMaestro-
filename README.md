# BioMaestro ML Project

## Current Project Structure

```
ML - BioMaestro/
│
├── .venv/                # Python virtual environment
│
├── data/
│   ├── preprocessed_spectrograms/
│   ├── unlabeled_corpus/
│   └── rfcx-species-audio-detection/
│       ├── sample_submission.csv
│       ├── train_fp.csv
│       ├── train_tp.csv
│       ├── test/
│       ├── tfrecords/
│       └── train/
│
├── lightning_logs/       # Training logs
│
├── models/              # Model directory
│   ├── biomaestro_checkpoints/
│   │   ├── biomaestro-epoch=00-train_loss=0.17.ckpt
│   │   ├── biomaestro-epoch=10-train_loss=0.07.ckpt
│   │   ├── biomaestro-epoch=18-train_loss=0.06.ckpt
│   │   ├── biomaestro-epoch=27-train_loss=0.06.ckpt
│   │   ├── last-v1.ckpt
│   │   └── last.ckpt
│   │
│   └── mae_checkpoints/
│       ├── last-v1.ckpt
│       ├── last-v2.ckpt
│       ├── last-v3.ckpt
│       ├── last-v4.ckpt
│       ├── last.ckpt
│       ├── rainforest-mae-epoch=00-train_loss=0.75.ckpt
│       ├── rainforest-mae-epoch=01-train_loss=0.75.ckpt
│       ├── rainforest-mae-epoch=02-train_loss=0.74.ckpt
│       ├── rainforest-mae-epoch=04-train_loss=0.75.ckpt
│       ├── rainforest-mae-epoch=06-train_loss=0.75.ckpt
│       └── ... (more checkpoint files)
│
│
├── src/
│   ├── build_unlabeled_corpus.py
│   ├── preprocess_data.py
│   ├── run_inference.py
│   ├── train_biomaestro.py
│   ├── train_mae.py
│   ├── utils.py
│   └── __pycache__/
│
└── requirements.txt    # Python dependencies
```

## Directory Descriptions

- `configs/`: Contains configuration files for model parameters, training settings, and hyperparameters
- `data/`: Raw and processed data files
- `models/`: Trained model files and checkpoints
- `notebooks/`: Jupyter notebooks for data exploration and experimental analysis
- `src/`: Main source code directory
  - `preprocessing/`: Scripts for data cleaning and feature engineering
  - `training/`: Model training and validation code
  - `utils/`: Helper functions and utility scripts
  - Main scripts:
    - `build_unlabeled_corpus.py`: Script for building unlabeled dataset
    - `preprocess_data.py`: Data preprocessing pipeline
    - `run_inference.py`: Model inference script