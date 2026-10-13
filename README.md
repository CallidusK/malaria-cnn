# Malaria Cell Classifier

*Made with ❤️ by Calli*

A convolutional neural network (CNN) that looks at a microscope image of a single red blood cell and says whether it is **Parasitized** (malaria-infected) or **Uninfected** (healthy).

Problem statement by **Biocon Ltd.**: support faster and more accurate automated diagnosis of blood samples.

> 🚧 **Work in progress.** This README will be completed in Phase 8, with real results from Kaggle training runs.

## Dataset

[NIH Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria), from the U.S. National Library of Medicine (Lister Hill National Center for Biomedical Communications).

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create a tiny fake dataset for local testing
python -m src.make_fake_data
```

Real training runs on Kaggle (instructions coming in Phase 4).

## Project plan

See [`PLAN.md`](PLAN.md) for the full plan and [`docs/learning_notes.md`](docs/learning_notes.md) for plain-English notes on every concept used.
