# Goodreads Shelf Prediction

Predicts a romance book's shelves from its description.

## Dataset

Download `goodreads_books_romance.json.gz` from the [Goodreads dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)

## Local modeling contribution

The original `code.py` is preserved. The reproducible extension is in `modeling/`.
It keeps the original 15-shelf task, adds a work-disjoint validation partition,
compares eight sparse linear models, tunes thresholds on validation only, and
runs an explicitly separate merged-label sensitivity experiment.

Start with the [English modeling guide](modeling/MODELING_GUIDE_EN.md) or the
[Chinese modeling guide](modeling/MODELING_GUIDE_ZH.md), or open
`modeling.html` for the executed English modeling notebook. This is the modeling
contribution to the group's shared workbook, not the complete five-section
assignment or the presentation video.

Use Python 3.11+ with `requirements-modeling.txt` installed:

```bash
python -m pip install -r requirements-modeling.txt
python -m scripts.download_romance
python -m unittest discover -s tests -v
python -m modeling.run --jobs 4
python -m scripts.build_modeling_report
python -m scripts.verify_modeling
python -m modeling.predict "A young woman meets a mysterious vampire in a haunted town."
```

On the machine used for the recorded run the dependency-equipped interpreter was
`/home/cauthor/anaconda3/bin/python3`; `/usr/bin/python3` did not have pandas.
Use the former in place of `python` if reproducing on that same machine.
The upstream `code.py` has the same name as Python's standard-library module.
The report builder handles this collision without changing that original file.
For an interactive Jupyter session, launch with `PYTHONSAFEPATH=1 jupyter lab`.
The character-feature comparison uses substantially more memory than word-only
models; reduce `--jobs` to 1 on a machine with less available RAM. The recorded
run used four parallel OvR classifiers on a machine with 32 GiB of RAM.

Read `modeling/HANDOFF.md` before merging with the group notebook. Tables and
figures live under `reports/full/`. Fitted model bundles and prediction arrays
live under `artifacts/full/` and are ignored by Git, as is the raw dataset.
The dataset is for academic use and should not be redistributed; see its
[official terms](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html).
To rerun the original exploration, set `GOODREADS_PATH` in a local `secret.env`
and install its additional `python-dotenv` dependency.
