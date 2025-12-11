## Where to download dataset

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Set up Virtual Environment 

Ensure to set up your virtual environment and install requirements

## How to train models

Step 1 - Preprocess Data

```bash
python training/preprocess.py
```

Step 2 - Train Models

```bash
python training/train_mlp.py
```
```bash
python training/train_lstm.py --augment-only
```

Step 3 - Export to TensorFlow.js

```bash
python training/export_models.py
```

Step 4 - Copy to frontend

```bash
cp -r models/tfjs/* frontend/public/models/
```

## Running Frontend
```bash
pnpm install
```
```bash
pnpm run dev
```