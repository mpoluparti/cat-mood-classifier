# Cat Mood Classifier

Classify cat meow audio clips by the context they were recorded in
(brushing, waiting for food, isolation), as a proxy for the cat's
emotional/behavioral state.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

This project expects the [CatMeows dataset](https://www.kaggle.com/datasets/andrewmvd/cat-meows)
(or equivalent) extracted into `data/raw/`. Filenames are expected to start
with a context code:

- `B_` = brushing
- `F_` = waiting for food
- `I_` = isolation in an unfamiliar environment

Check the actual filenames after downloading -- `src/data_utils.py` assumes
this convention and will silently skip files that don't match it.

## Usage

Train and evaluate the primary model (hand-crafted features + SMOTE):

```bash
python -m src.train
```

Predict on a new clip:

```bash
python -m src.predict path/to/clip.wav
```

Run the YAMNet embeddings experiment (currently underperforms the
primary model, kept for reference):

```bash
python -m src.train_embeddings
```

## Web app

A FastAPI backend serves the trained model, and a React (Vite) frontend
lets you upload a clip and see the predicted mood with per-class
probabilities.

Backend (from the project root, with the venv active):

```bash
uvicorn backend.main:app --reload
```

Frontend (in a separate terminal):

```bash
cd frontend
npm install
npm run dev
```

Then open the Vite dev server URL (usually http://localhost:5173).

## Project layout

```
data/raw/        raw downloaded audio (not committed)
data/processed/  cached extracted features (not committed)
src/
  data_utils.py       scan data/raw and build a labeled file index
  preprocess.py       load, trim silence, normalize, fix clip length
  features.py         extract MFCC / pitch / spectral / energy features
  train.py            build dataset, train RandomForest with SMOTE, save it
  predict.py          load the saved model and classify a single clip
  embeddings.py        extract pretrained YAMNet embeddings
  train_embeddings.py  train/compare classifiers on YAMNet embeddings
backend/
  main.py          FastAPI app: POST /predict (audio -> label + probabilities)
frontend/          React (Vite) upload UI, calls the backend API
models/          saved trained models (classifier.joblib is committed)
```

## Results

Primary model (RandomForest on MFCC/pitch/spectral/energy features,
SMOTE-balanced training set), on a held-out 20% test split of 440 clips:

| class | precision | recall | f1 |
|---|---|---|---|
| brushing | 0.75 | 0.69 | 0.72 |
| isolation | 0.82 | 0.95 | 0.88 |
| waiting_for_food | 0.69 | 0.50 | 0.58 |
| **overall accuracy** | | | **0.78** |

YAMNet embeddings were tried as an alternative feature source but scored
lower (55-57% test accuracy) -- see `src/train_embeddings.py`.

## Status / next steps

- [x] Project scaffolding
- [x] Baseline hand-crafted-feature + RandomForest pipeline
- [x] Fix class imbalance (SMOTE) -- 67% -> 78% test accuracy
- [x] Try YAMNet embeddings -- underperformed, kept as a documented experiment
- [x] Web app: FastAPI backend + React frontend for uploading a clip and getting a prediction
- [ ] Try hyperparameter tuning / Gradient Boosting on the current features
- [ ] Optional: CNN on spectrograms (needs more data to avoid overfitting)
- [ ] Optional: deploy the web app (e.g. Render/Fly.io for the API, Vercel/Netlify for the frontend)
