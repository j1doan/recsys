Plan: A2, predict Goodreads shelves from book description

*Task*
- Shelf = tag users put book on (to-read, m-m, paranormal). Book has list of shelf name + user count.
- Input: description text. Output: which sub-genre shelves book has. Multi-label classification.
- Labels already in data. No hand labelling. Rule turns counts into 0/1.

*Data*
- Goodreads romance books, 335,449 rows, metadata only. No users, no reviews. So no LFM/BPR.
- 304,837 have description (91%).
- Dirty: publication_year min 1, max 65535. average_rating min 0. 0.4% books have 0 ratings.
- Median 52 ratings per book, mean 448, max 2M. Heavy tail.
- 84% of similar_books links point inside file (earlier 10% on first 40k rows was wrong guess). Link prediction (option C) now viable.

*Label rule*
- Drop non-content shelves (to-read, favorites, ebook, owned, romance itself...).
- Book needs >= 10 content votes. Shelf is label if >= 10% of book content votes. Top 15 shelves.
- 299,789 books kept. 0.44 labels per book (most books have none of the 15).
- Sweep: at 5% cutoff 15th shelf has 10,545 books, at 10% 3,105, at 20% 376. 10% ok. 20%+ too thin.

*Eval*
- Split by work_id (editions stay together). 239,379 train, 60,410 test.
- Metrics: micro-F1, macro-F1, macro-AUC, per-shelf F1 + AUC. No accuracy (most labels negative).

*Results (full file, laptop CPU, 3.6 min)*
- Prior baseline: micro-F1 0.000
- TF-IDF + one-vs-rest logistic regression: micro-F1 0.464, macro-F1 0.443, macro-AUC 0.930
- Best: manga F1 0.739 AUC 0.992, historical-romance F1 0.653, m-m F1 0.584
- Worst: contemporary F1 0.237, contemporary-romance 0.268
- AUC high everywhere (0.82 to 0.99), F1 low. Ranking fine, threshold at 0 not tuned.

*Problems seen*
- Near-duplicate shelves: historical / historical-romance, contemporary / contemporary-romance, paranormal / paranormal-romance. Merge or keep? Contemporary pair is the worst scored.
- Shelf "manga" odd for romance, but easiest to predict (distinct text).
- Most books have no label. Test set mostly negatives.

*Next*
- Decide merge of near-duplicate shelves.
- Tune threshold per shelf (AUC much better than F1).
- Keyword baseline (shelf name in description).
- Stronger model if time (tune C, char n-grams, small embeddings).
- EDA plots, error examples read by hand.
- Related work: Goodreads dataset papers (Wan and McAuley), text classification. Cite from real sources, not memory.

*Presentation (~20 min)*
1. Task, eval, baselines, validity (5)
2. Data context, preprocessing, EDA (4)
3. Modeling (4)
4. Evaluation, per-shelf, errors (4)
5. Related work (3)

*Deliverables*
- workbook.html (clean notebook export, starts with DOCTYPE html)
- video_url.txt (Drive link or YouTube, mp4 > 1 MB)
- responses.txt for peer grading later
- Deadline: Tue 12/1 (check exact time on gradescope)

*Schedule*
- Wed 11/25: decide shelf merge + threshold tuning. Keyword baseline.
- Thu 11/26 (Thanksgiving): improve model if time. EDA plots. Light day.
- Fri 11/27: error analysis. Related work: find real papers, read.
- Sat 11/28: clean notebook top to bottom, run fresh kernel. Draft slides.
- Sun 11/29: DONE day. Notebook final, export workbook.html, slides final, rehearse once.
- Mon 11/30: record video (~20 min), upload, check link plays logged out. Submit early.
- Tue 12/1: buffer only. Submit by gradescope time.
- Wed 12/2: peer grading opens. Upload empty responses.txt to get assignments. Watch assigned videos, submit responses.txt (about 1 week, 4 marks). Check due date on gradescope.

*Group (up to 4, same grade for all)*
- Split by work area, names TODO:
  - Modeling: shelf merge, threshold tuning, stronger model
  - Eval: keyword baseline, per-shelf results, error analysis
  - EDA + data context: plots, cleaning notes
  - Related work + slides + video
- One shared notebook, one owner merges. Fresh-kernel run before export.
- Video: everyone talks a section, or one person records. Decide by Sat.
- Each member does own peer grading? Check gradescope on 12/2.

*Open (need decision)*
- Group size and who takes what
- Merge near-duplicate shelves or not
- Threshold tuning approach
