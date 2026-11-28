# Goodreads Shelf Prediction

Predicts a romance book's Goodreads shelves from its description.

## Setup

Download `goodreads_books_romance.json.gz` from the [Goodreads dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) and set `GOODREADS_PATH` in `secret.env`.

```bash
pip install numpy pandas scipy matplotlib scikit-learn torch python-dotenv jupytext nbconvert ipykernel
```

## Run

```bash
python -m jupytext --to ipynb --execute code.py
python -m nbconvert --to html code.ipynb
```
