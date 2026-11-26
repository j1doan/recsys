"""Download the official romance metadata atomically; verify gzip integrity.

The dataset is local-only and is excluded by .gitignore. No interaction/review
datasets are needed for the description-only modeling task.
"""
import argparse
import gzip
import json
from pathlib import Path
from urllib.request import Request, urlopen

from modeling.core import DATA_URL, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=Path('data/goodreads_books_romance.json.gz'))
    args = parser.parse_args()
    dest = args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_suffix(dest.suffix+'.part')
        req = Request(DATA_URL, headers={'User-Agent':'GoodreadsCourseProject/1.0'})
        with urlopen(req, timeout=60) as response, part.open('wb') as f:
            while chunk := response.read(8*1024*1024):
                f.write(chunk)
        with gzip.open(part, 'rb') as f:
            while f.read(8*1024*1024):
                pass
        part.replace(dest)
    with gzip.open(dest, 'rt', encoding='utf-8') as f:
        first = json.loads(next(f))
    if not {'book_id','description','popular_shelves','work_id'} <= first.keys():
        raise ValueError('Unexpected dataset schema')
    print(f'{dest}: {dest.stat().st_size:,} bytes; sha256={sha256_file(dest)}')


if __name__ == '__main__':
    main()
