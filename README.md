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

Train and evaluate:

```bash
python -m src.train
```

Predict on a new clip:

```bash
python -m src.predict path/to/clip.wav
```

## Project layout

```
data/raw/        raw downloaded audio (not committed)
data/processed/  cached intermediate data (not committed)
src/
  data_utils.py  scan data/raw and build a labeled file index
  preprocess.py  load, trim silence, normalize, fix clip length
  features.py    extract MFCC / pitch / spectral / energy features
  train.py       build dataset, train + evaluate a RandomForest, save it
  predict.py     load the saved model and classify a single clip
models/          saved trained model (not committed)
```

## Status / next steps

- [x] Project scaffolding
- [ ] Download dataset into `data/raw/`
- [ ] Confirm filename convention matches `data_utils.py`
- [ ] Run `python -m src.train`, check class balance and confusion matrix
- [ ] Iterate on features/model if baseline accuracy is weak
- [ ] Optional: CNN on spectrograms or transfer-learned embeddings (YAMNet/PANNs)
- [ ] Optional: simple CLI/web demo for uploading a clip and getting a prediction
