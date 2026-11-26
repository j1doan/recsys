"""Build a fresh-kernel modeling notebook, self-contained HTML and local guides.

The expensive full-data training is performed by modeling.run. This notebook
reads its recorded reports and fitted model, showing all outputs to peer graders.
"""
from __future__ import annotations

import json
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/recsys-mpl')
os.environ.setdefault('IPYTHONDIR', '/tmp/recsys-ipython')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/recsys-jupyter')
os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
# The untouched upstream code.py shadows Python's stdlib code module, which
# IPython/pdb imports during kernel startup. Preload the stdlib in this process
# and suppress implicit cwd insertion in the fresh child kernel (Python >=3.11).
os.environ['PYTHONSAFEPATH'] = '1'
_spec = importlib.util.spec_from_file_location('code', Path(sysconfig.get_path('stdlib'))/'code.py')
_stdlib_code = importlib.util.module_from_spec(_spec)
sys.modules['code'] = _stdlib_code
_spec.loader.exec_module(_stdlib_code)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import joblib
import nbformat as nbf
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'reports/full'


def read_json(name):
    return json.loads((REPORT/name).read_text())


def md_table(df, digits=4):
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append([f'{x:.{digits}f}' if isinstance(x, (float, np.floating))
                     else str(x) for x in row])
    return '\n'.join(['| ' + ' | '.join(map(str,df.columns))+' |',
                      '| '+' | '.join(['---']*len(df.columns))+' |',
                      *['| '+' | '.join(row)+' |' for row in rows]])


def figures(selection, search, test):
    dest = REPORT/'figures'
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':140})
    fig, ax = plt.subplots(figsize=(12,3.1))
    ax.set(xlim=(0,12),ylim=(0,3));ax.axis('off')
    boxes=[(.1,'Book description\n(text only)'),(3.15,'TF-IDF\nfit on train only'),
           (6.2,'15 binary classifiers\nfit weights on train'),
           (9.25,'15 score thresholds\nselect on validation')]
    for x, label in boxes:
        ax.add_patch(FancyBboxPatch((x,1.4),2.5,.85,
            boxstyle='round,pad=0.08',facecolor='#eaf1f5',edgecolor='#236b8e'))
        ax.text(x+1.25,1.825,label,ha='center',va='center',fontsize=10)
    for x in (2.65,5.7,8.75):
        ax.annotate('',(x+.4,1.825),(x,1.825),arrowprops={'arrowstyle':'->','color':'#236b8e'})
    ax.text(6,.75,'Shelf votes create training targets; work IDs define disjoint splits.',
            ha='center',va='center',fontsize=10,color='#465867')
    ax.text(6,.25,'Freeze the model and thresholds before computing test metrics.',
            ha='center',va='center',fontsize=11,color='#965d1d')
    fig.tight_layout();fig.savefig(dest/'modeling_pipeline.png');plt.close(fig)
    pivot = search.pivot(index='candidate', columns='policy', values='macro_f1')
    pivot = pivot.sort_values('per_label')
    fig, ax = plt.subplots(figsize=(10,5.5))
    yy = np.arange(len(pivot))
    ax.barh(yy-.18, pivot['default'],height=.34,label='Default threshold',color='#a7b6c5')
    ax.barh(yy+.18,pivot['per_label'],height=.34,label='Per-label validation thresholds',color='#236b8e')
    ax.set(yticks=yy,yticklabels=pivot.index,xlim=(0,1),xlabel='Validation macro-F1',
           title='Model selection uses validation data only')
    ax.legend(loc='lower right'); fig.tight_layout()
    fig.savefig(dest/'validation_search.png'); plt.close(fig)

    main = test[(test.task=='original_15') & test.policy.isin(['default','per_label'])].copy()
    main['model'] = main.candidate+' / '+main.policy
    fig, ax = plt.subplots(figsize=(10,4.2))
    yy=np.arange(len(main))
    ax.barh(yy-.18,main.micro_f1,.34,label='Micro-F1',color='#236b8e')
    ax.barh(yy+.18,main.macro_f1,.34,label='Macro-F1',color='#e69b44')
    ax.set(yticks=yy,yticklabels=main.model,xlim=(0,1),xlabel='Held-out test F1',
           title='Same task and same training partition for all comparisons')
    ax.legend(loc='lower right');fig.tight_layout()
    fig.savefig(dest/'test_ablation.png');plt.close(fig)

    ref = pd.read_csv(REPORT/'per_label_word_lr_balanced_C1.csv').set_index('shelf')
    selected = pd.read_csv(REPORT/f'per_label_{selection["selected_candidate"]}.csv').set_index('shelf')
    selected=selected.sort_values('f1')
    fig,ax=plt.subplots(figsize=(9,6.5))
    yy=np.arange(len(selected))
    ax.barh(yy-.17,ref.loc[selected.index,'f1'],.32,label='Word LR, tuned thresholds',color='#a7b6c5')
    ax.barh(yy+.17,selected.f1,.32,label='Selected model, tuned thresholds',color='#236b8e')
    ax.set(yticks=yy,yticklabels=selected.index,xlim=(0,1),xlabel='Test F1',
           title='Per-shelf performance, original 15 labels')
    ax.legend(loc='lower right');fig.tight_layout()
    fig.savefig(dest/'per_shelf_f1.png');plt.close(fig)


def guides(audit, selection, test, manifest, bootstrap, unseen):
    main=test[test.task=='original_15']
    selected=main[(main.candidate==selection['selected_candidate']) & (main.policy=='per_label')].iloc[0]
    base=main[(main.candidate=='word_lr_balanced_C1') & (main.policy=='default')].iloc[0]
    tuned=main[(main.candidate=='word_lr_balanced_C1') & (main.policy=='per_label')].iloc[0]
    global_selected=main[(main.candidate==selection['selected_candidate']) & (main.policy=='global')].iloc[0]
    merged=test[test.task!='original_15'].iloc[0]
    ci=bootstrap['vs_reference_default']['delta_macro_f1']
    table=md_table(main[['candidate','policy','micro_f1','macro_f1','macro_auc','macro_ap']])
    validation=pd.read_csv(REPORT/'validation_search.csv')
    efficiency=validation[validation.policy=='per_label'][
        ['candidate','macro_f1','train_seconds','feature_seconds','n_features']]
    efficiency_table=md_table(efficiency, digits=3)
    chinese=f'''# Goodreads romance modeling 中文说明

## 这份作业要做什么

课程 Assignment 2 要求用真实数据完成一个预测项目，解释任务、数据、模型、评估和相关工作，
提交可直接阅读的 notebook HTML，并录制约 20 分钟的小组讲解。你负责其中约 4 分钟的 modeling。
原始要求见仓库上一级目录的作业 PDF；小组分工见 `PROJECT_PLAN.md`。

本组的具体问题是：**输入 romance 子集中一本书的简介，预测读者给它的内容书架标签。**
例如一本书既可能是 historical，也可能是 historical-romance；模型要对每个标签分别判断。
所以这是多标签文本分类：不是评分回归，也不是为某个用户推荐下一本书。
这份元数据没有用户逐本交互矩阵，直接使用 BPR 或用户/物品矩阵分解不符合当前输入。

## 标签从哪里来

Goodreads 的 `popular_shelves` 提供书架名和投票数。保留仓库原来的排除词表：去掉 to-read、
owned、ebook、romance、fiction 等状态、格式和宽泛标签。剩余投票总数至少为 10，且简介非空，
才保留这本书。对某标签，若其票数占该书全部剩余票数的至少 10%，标签为 1，否则为 0。
这沿用了组内的启发式规则；排除词表并不能完全过滤所有非内容书架。

例如剩余票数为 100，historical 有 14 票，则其标签为 1；只有 9 票则为 0。
从训练集里选择正例最多的 15 个标签，并在验证/测试时固定。
**0 代表不满足这个规则，不代表人工确认这本书绝对没有这种内容。**
零标签图书也保留，因为强行每本书选一个标签会改变任务。
`manga` 更像媒介形式，也出现在原始的前 15 个书架中，因此本次继续保留；
预测目标应准确称为“读者书架标签”，并不是一个严格、统一的爱情子类型分类体系。

全量 {audit['raw_books']:,} 本，简介非空 {audit['nonempty_descriptions']:,} 本，
最终保留 {audit['eligible_books']:,} 本。平均每本 {audit['mean_labels_per_book']:.3f} 个标签，
{audit['zero_label_fraction']:.1%} 的书没有这 15 个标签中的任何一个。

## 三个集合怎样分

按 `work_id` 分组，同一作品的不同版本必须一起进入同一个集合。
保留小组 seed=0 的外层测试划分，再从原训练部分按 seed=1 留出验证集：

- 训练集：{audit['split_sizes']['train']:,} 本，用于学习词表、IDF 和模型权重。
- 验证集：{audit['split_sizes']['validation']:,} 本，用于选算法、C 和每个标签的阈值。
- 测试集：{audit['split_sizes']['test']:,} 本，模型选定之后才计算最终结果。

三个集合的作品重叠为 0。模型不在选型后用训练+验证重新拟合，因为这样会改变分数尺度，
原验证阈值不一定仍适用。`selection.json` 在测试计算之前写出。
计划里之前约 0.443 的 macro-F1 来自更多训练书和早期流程；下表是本次统一划分的内部对照，
不能把两个流程的数字差异都归于模型改进。

## 为什么选这些算法

**TF-IDF** 把简介变成一个稀疏向量。常见于某本书、但不是所有书都在说的词获得较高权重。
词 unigram 和 bigram 能保留 vampire、duke、high school 等词及短语。
词表最多 50,000 维，至少在训练集 5 本书中出现；只在训练集拟合。
不输入书架名/票数、评分、作者或标题，避免直接把答案当输入。

**One-vs-rest 逻辑回归** 为 15 个标签各训练一个二分类器。
每个模型学习一个线性分数，sigmoid 将其转成加权训练目标下的概率形式；
`class_weight=balanced` 提高少数类的损失权重，但这些输出不能直接当作真实发生率。
优点是适合稀疏文本、训练快、词权重容易解释；缺点是难理解长距离语义，而且独立标签模型不显式建模标签依赖。

**线性 SVM** 同样为每个标签学习一个线性函数，但优化平方 hinge loss，鼓励正负例之间留出间隔。
它适合高维文本；输出是分类分数，不是概率。使用 LinearSVC，不建立不可扩展的核矩阵。
逻辑回归与 SVM 都有 L2 正则；C 越大，相对越强调拟合训练误差，正则越弱。

**词特征加字符特征**：增加最多 60,000 个词内 3–5 字符片段。
词向量权重为 1，字符向量为 0.5；两个块各自先做 L2 归一化，再拼接。
字符片段可能补充词形、拼写和非英语线索，但会增加内存和时间。
这是一种待验证的改进，而不是默认保证更强。
本次最多 {audit['split_sizes']['train']:,} × {int(efficiency.n_features.max()):,} 个特征值，
若存成 float64 稠密矩阵约需
{audit['split_sizes']['train']*int(efficiency.n_features.max())*8/1024**3:.1f} GiB。
因此实现一直保留稀疏矩阵，并限制同时训练的分类器数量。

本次预先定义了 8 个候选：词 LR 的 balanced C=0.25/1/4、未加权 C=1，
词 SVM 的 C=0.25/1，以及词+字符 LR/SVM 的 C=1。
按验证集逐标签阈值下的 macro-F1 选型，micro-F1 作为并列时的次级指标。
最终选中 **{selection['selected_candidate']}**，验证 macro-F1 为
**{selection['selected_validation_macro_f1']:.4f}**。

下表是在验证集选型时测量的效率。`train_seconds` 包含训练和一次验证打分，
`feature_seconds` 是同一种特征方案共享的训练特征拟合与验证变换开销，不应对每个候选重复相加。

{efficiency_table}

## 阈值调优到底做了什么

模型先输出分数 s，之后才决定是否预测这个标签。仓库原来默认是 s > 0。
本实验统一使用 s >= 阈值，阈值与相同分数的处理方式在训练、验证和预测中保持一致。

1. 默认阈值：全部标签用 0。
2. 全局阈值：在验证集所有书-标签对上最大化 micro-F1。
3. 逐标签阈值：每个标签分别在验证集最大化其 F1，因而直接对应 macro-F1 的目标。

实现将验证分数排序，用累计 TP 计算每个不同分数边界的 F1，复杂度每标签 O(n log n)。
相同分数整组处理，F1 并列取较高阈值；没有正例时预测为空。
阈值只改变 0/1 判定，不改变同一模型分数的排序，所以同一模型的 ROC-AUC 和 AP 不变。
这里没有用测试集找阈值，也没有每本书强制取 top-1。

## 全量实测结果

{table}

相同训练集下，词 LR 从默认阈值到逐标签阈值，macro-F1 从 {base.macro_f1:.4f}
变为 {tuned.macro_f1:.4f}；最终模型达到 micro-F1 **{selected.micro_f1:.4f}**、
macro-F1 **{selected.macro_f1:.4f}**、macro-AUC **{selected.macro_auc:.4f}**。
最终模型相对默认词 LR 的 macro-F1 绝对差为 {selected.macro_f1-base.macro_f1:.4f}；
按作品进行 300 次配对 bootstrap 的 95% 区间为 [{ci['low']:.4f}, {ci['high']:.4f}]。
这个区间反映测试作品抽样的不确定性，不包含重复训练和重新选型的变异。

也要看到取舍：同一最终模型使用全局阈值时，测试 micro-F1 为
{global_selected.micro_f1:.4f}，高于逐标签方案；macro-F1 为 {global_selected.macro_f1:.4f}。
主配置仍遵循测试前确定的验证集 macro-F1 选型规则，不能把逐标签阈值说成对所有指标都更优。

micro-F1 汇总所有标签的 TP/FP/FN，常见标签影响更大；macro-F1 是各标签 F1 的平均，
更能反映稀有标签。AUC 衡量正例分数是否高于负例，不直接衡量最终标签集合是否正确。
另外报告 AP，因为大量负例下只看 ROC-AUC 容易高估使用体验。不以 accuracy 作为主要指标，
否则大量预测“都没有”也可能看起来很高。

## 近义标签最后怎样处理

主任务保留 15 个标签，便于和组内现有分析对接。historical 和 historical-romance 等
并不完全同义，所以另做 12 标签敏感性实验：合并 historical、contemporary、paranormal 三对。
**先按原规则生成各自的二值标签，再做 OR，不相加投票。**同一读者可能用过两个书架，
直接加票数并重新套 10% 规则会隐含重复计数；OR 避免这个额外假设。

12 标签实验沿用选定模型结构，重新用训练集拟合，并在验证集重新定阈值。
测试 micro-F1={merged.micro_f1:.4f}、macro-F1={merged.macro_f1:.4f}。
由于预测目标变宽了，这个分数不能作为原 15 标签任务上的增益。
训练集的标签交并比见 `reports/full/alias_analysis.csv`。

## 有哪些真实限制

虽然作品分组不重叠，但测试集中仍有 {audit['exact_description_overlap']['test_books_with_text_in_train_or_validation']}
本书的规范化简介与训练或验证集相同。额外报告了去掉这些书的测试子集：
{unseen['test_books']:,} 本，最终 macro-F1={unseen['selected']['macro_f1']:.4f}。
这只是诊断，不等于按全部重复简介/系列/作者重新分组训练；近似重复也没有完全消除。

书架投票是有噪声的弱标签；10% 和至少 10 票是小组设定的任务定义，未由人工标注验证。
作者和系列可能跨集合，测试不代表新作者泛化；一次分组划分也不能证明跨年份或跨平台泛化。
词袋模型缺少语义和标签关系建模，逐标签阈值也可能对小标签过拟合验证集。

## 你可以怎样讲这四分钟

- 0:00–0:40：简介输入、15 维 0/1 输出、多标签；用一个标签票数比例的例子解释监督信号。
- 0:40–1:35：TF-IDF、15 个独立逻辑回归、类别不平衡和 L2 正则，展示 `make_features` 和 `classifier`。
- 1:35–2:30：默认阈值与逐标签阈值不同；展示 `best_f1_threshold`，强调只用验证集。
- 2:30–3:20：讲解验证集比较图和最终模型，解释字符特征/线性 SVM 的时间与效果取舍。
- 3:20–4:00：说明标签合并是独立实验，展示一张结果表，并说明弱标签与重复简介限制。

建议先自己运行一次预测、查看一个标签的前十个词权重，并阅读一个误判案例，
用自己理解的措辞讲解，确保代码和陈述一致。`error_review_queue.csv` 是待人工阅读的队列，
不应把自动选出的例子当成人工确认的错误原因。

## 本地文件和复现

- `modeling.ipynb` / `modeling.html`：已在全新 kernel 执行的英文 modeling 章节，含输出与图。
- `modeling/core.py`：数据、标签、分组、特征、阈值、指标。
- `modeling/run.py`：8 模型实验、选择锁定、测试、合并实验与 bootstrap。
- `artifacts/full/selected_model.joblib`：可直接预测的已训练模型；只载入自己信任的本地 joblib。
- `reports/full/`：CSV、JSON、PNG 和复现记录；耗时 {manifest['wall_seconds']/60:.1f} 分钟。
- `modeling/HANDOFF.md`：给组内 notebook 合并负责人的接口与注意事项。

在仓库根目录运行 README 的命令。此机器使用
`/home/cauthor/anaconda3/bin/python3`，系统 `/usr/bin/python3` 缺少所需包。
原始数据和训练产物已加入 `.gitignore`，原 `code.py` 未改动。
原 `code.py` 与 Python 标准库同名，可能干扰 IPython。导出脚本已经处理此问题；
手动打开 Jupyter 时用 `PYTHONSAFEPATH=1 jupyter lab` 启动，并使用 Python 3.11+。

## 资料

- [Goodreads 官方数据说明](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)：数据来源和使用条件。
- [scikit-learn TF-IDF](https://scikit-learn.org/1.6/modules/feature_extraction.html#text-feature-extraction)。
- [逻辑回归](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression)。
- [线性 SVM](https://scikit-learn.org/1.6/modules/svm.html#classification)。
- [分类阈值调优](https://scikit-learn.org/1.6/modules/classification_threshold.html)：分类分数与判定阈值的区别。
- Wan and McAuley (2018), [Item Recommendation on Monotonic Behavior Chains](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf)。
- Wan et al. (2019), [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248/)。

后两篇是数据相关文献，分别研究用户行为链推荐和评论剧透检测，任务与本项目不同，
其论文分数不能直接与我们的书架分类 F1 比较。
'''
    (ROOT/'modeling/MODELING_GUIDE_ZH.md').write_text(chinese,encoding='utf-8')
    english=f'''# Goodreads Romance Modeling Guide

## What this assignment and our modeling section cover

Assignment 2 asks the group to build a prediction project using real data and
explain the task, data, models, evaluation, and related work. The group submits
a readable notebook exported as HTML and an approximately 20-minute presentation
video, followed by peer grading. Our group plan allocates approximately four
minutes to modeling. The original assignment PDF is one directory above this
repository in the local workspace; the group plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

**Our task is to predict a book's reader-assigned shelf labels from its description
in the Goodreads romance subset.** A book can belong to several shelves, such as
`historical` and `historical-romance`, or to none of the selected shelves.
This is a multi-label text classification problem.

The input is description text only. The output is a 15-element binary vector,
with one prediction for each shelf. This metadata file does not contain a
per-user interaction matrix, so user-item matrix factorization and BPR do not
directly match the current task.

This guide covers the modeling contribution and its supporting experiments.
The group still needs to integrate the other sections into the shared workbook
and record the presentation.

## How the labels are constructed

Goodreads provides shelf names and counts in `popular_shelves`. We retain the
original repository's exclusion list, removing shelves such as `to-read`,
`owned`, `ebook`, `romance`, and `fiction`. A book is eligible when it has a
nonempty description and at least 10 remaining shelf votes.

For an eligible book, a shelf is positive when its count is at least 10% of
the book's total remaining shelf votes. For example, if a book has 100 remaining
votes, 14 votes for `historical` make that label positive; 9 votes do not.
The 15 shelves with the most positive training examples form the fixed label
vocabulary for validation and test.

These are the group's heuristic labels. The inherited exclusion list does not
remove every non-content shelf. **A zero means the shelf did not pass the rule,
not that a human confirmed the book lacks that content.** We retain books with
no positive target labels; forcing a label onto every book would change the task.

We also retain `manga`, which appeared among the original top 15 shelves. It
describes a medium rather than a romance subgenre, so the target should be
described as reader shelf usage rather than a uniform genre taxonomy.

The complete file contains {audit['raw_books']:,} books, of which
{audit['nonempty_descriptions']:,} have nonempty descriptions. The filtering rule
retains **{audit['eligible_books']:,} books**. Each book has an average of
{audit['mean_labels_per_book']:.3f} positive target labels, and
{audit['zero_label_fraction']:.1%} have none of the 15 selected labels.

## Training, validation, and test partitions

We split by `work_id`, keeping editions of the same work in one partition.
The outer split preserves the group's original test book IDs using seed 0.
An inner grouped split with seed 1 reserves validation data from the original
development partition.

| Partition | Books | Purpose |
| --- | --- | --- |
| Training | {audit['split_sizes']['train']:,} | Fit the vocabulary, IDF, and classifier weights |
| Validation | {audit['split_sizes']['validation']:,} | Select the model, C, and thresholds |
| Test | {audit['split_sizes']['test']:,} | Evaluate after model selection is frozen |

There is no work-group overlap between partitions. We select the label vocabulary
and fit both TF-IDF representations using training data only. The choice is
written to `reports/full/selection.json` before test scoring.

We do not refit the selected classifier on training plus validation data. Such a
refit could change its score scale and invalidate the validation-tuned thresholds.

The earlier plan's approximately 0.443 macro-F1 came from the original two-way
split, which used more books for training. Use the matched baselines from this
run to assess improvements; differences between the old and new workflows cannot
all be attributed to the model.

## Why these models are appropriate

### Word TF-IDF

TF-IDF represents each description as a sparse vector. Terms receive more weight
when they are frequent in a particular description but less common across books.
Word unigrams and bigrams retain individual words and phrases such as `vampire`,
`duke`, and `high school`.

The word representation uses at most 50,000 features, a minimum document frequency
of 5, English stop-word removal, sublinear term frequency, and L2 normalization.
The vocabulary and IDF are fitted on training descriptions only. Shelf fields,
ratings, author IDs, and titles are not model inputs. Work IDs are used for
splitting, and shelf counts are used to construct targets.

### One-vs-rest logistic regression

One-vs-rest trains one binary classifier for each of the 15 shelves. Each learns
a linear score from the same description vector. The sigmoid converts a score
to a probability-shaped output under the training objective. With
`class_weight='balanced'`, minority-class examples receive larger loss weights;
these outputs should not be treated as calibrated population probabilities.

Logistic regression works efficiently with sparse text and exposes interpretable
feature weights. Its limitations include limited understanding of long-range
meaning and the absence of an explicit model of dependencies between labels.

### Linear SVM

Linear SVM also learns one linear function per shelf, using squared hinge loss
to encourage separation between positive and negative examples. It is suitable
for high-dimensional sparse text. Its outputs are decision scores rather than
probabilities. We use `LinearSVC`, which avoids constructing a large nonlinear
kernel matrix.

### Training objectives and model selection

For label j, let `s_ij = w_j^T x_i + b_j`, `z_ij = 2*y_ij - 1`, and `a_ij` be
the training class weight. With the default intercept scaling of 1, liblinear
also regularizes the intercept; `w_tilde_j` includes both the coefficients and
the intercept. The objectives are:

```text
LR:  0.5 * ||w_tilde_j||^2 + C * sum_i a_ij * log(1 + exp(-z_ij * s_ij))
SVM: 0.5 * ||w_tilde_j||^2 + C * sum_i a_ij * max(0, 1 - z_ij * s_ij)^2
```

For balanced weighting, `a_ij = n / (2 * n_class)` within each training binary
problem. Both models use L2 regularization. A larger C weakens regularization
relative to fitting the training data.

**Training minimizes these regularized losses. Model and threshold selection use
validation F1.** The classifier does not directly optimize F1 during fitting.

### Adding character features

The richer representation adds at most 60,000 within-word character n-grams of
length 3 to 5 using `char_wb`. Word and character blocks are L2-normalized
separately and concatenated with fixed weights of 1.0 and 0.5, respectively.
Character fragments can capture morphology, spellings, names, and non-English
text, while increasing memory and computation.

The largest training feature matrix has {audit['split_sizes']['train']:,} rows
and {int(efficiency.n_features.max()):,} columns. A dense float64 version would
require approximately
{audit['split_sizes']['train']*int(efficiency.n_features.max())*8/1024**3:.1f} GiB.
The implementation therefore retains sparse matrices and limits parallel
classifiers. Prediction costs approximately `O(K * nnz(X))`, where K is the
number of labels and `nnz(X)` is the number of nonzero input features.

### Candidate comparison and measured efficiency

We declared eight configurations: balanced word LR with C=0.25, 1, or 4;
unweighted word LR with C=1; balanced word SVM with C=0.25 or 1; and balanced
word-plus-character LR and SVM with C=1.

Selection uses validation macro-F1 with per-label thresholds. Ties are broken
by validation micro-F1 and then candidate name. The selected candidate is
**`{selection['selected_candidate']}`**, with validation macro-F1
**{selection['selected_validation_macro_f1']:.4f}**.

{efficiency_table}

`macro_f1` in this table is measured on validation data. `train_seconds` includes
training and one validation scoring pass. `feature_seconds` is the shared cost
of fitting training features and transforming validation text for a representation;
it should not be charged separately to every candidate using that representation.
Runtime depends on the machine and parallelism. These results support the
selected configuration within the tested grid, rather than a claim that one
classifier is always better for text classification.

## What threshold tuning does

The model produces a score before making a binary decision. The original
repository predicted a label when its score was greater than zero. This
experiment consistently predicts a label when `score >= threshold`, including
when scores are tied.

We compare three policies:

1. **Default:** use zero for every label.
2. **Global:** choose one threshold maximizing validation micro-F1 across all
   book-label pairs.
3. **Per-label:** choose one threshold per shelf maximizing its validation F1.
   Independent per-label optimization corresponds to the macro-F1 objective.

The exact search sorts validation scores and uses cumulative true positives to
compute F1 at every distinct score boundary. All examples with the same score
are processed together. Ties in F1 keep the higher threshold; a shelf with no
validation positives predicts none. The cost is `O(n log n)` per label.

Thresholds change the binary predictions, but not the fitted scores or their
ranking. ROC-AUC and average precision therefore remain the same across
threshold policies for a given model. We do not tune thresholds on test data
or force every book to receive a top-1 label.

## Full-data results and interpretation

The table below evaluates the original 15-label task on the held-out test set.
The prior baseline thresholds training prevalences at 0.5. The keyword baseline
uses literal shelf-name matches after simple punctuation and space normalization.

{table}

On the same training partition, tuning thresholds increases word LR macro-F1
from {base.macro_f1:.4f} to {tuned.macro_f1:.4f}. The selected model achieves
**micro-F1 {selected.micro_f1:.4f}, macro-F1 {selected.macro_f1:.4f}, and
macro-AUC {selected.macro_auc:.4f}**.

Relative to default-threshold word LR, the macro-F1 gain is
{selected.macro_f1-base.macro_f1:.4f}, or approximately
{100*(selected.macro_f1-base.macro_f1):.2f} percentage points and
{100*(selected.macro_f1-base.macro_f1)/base.macro_f1:.1f}% relative improvement.
Using {bootstrap['repetitions']} paired bootstrap replicates that resample test
work groups, the 95% interval for that gain is
[{ci['low']:.4f}, {ci['high']:.4f}]. This conditions on the fitted models and
captures test-work sampling variation, not variation from retraining or model
selection.

There is also an operating-point trade-off. For the selected classifier, the
global threshold achieves test micro-F1 {global_selected.micro_f1:.4f}, higher
than the per-label policy, with macro-F1 {global_selected.macro_f1:.4f}.
The primary configuration follows the validation macro-F1 criterion declared
before test evaluation. Per-label thresholds are not better for every metric.

Micro-F1 pools true positives, false positives, and false negatives across labels,
so frequent shelves have more influence. Macro-F1 averages each shelf's F1
equally. ROC-AUC evaluates ranking, rather than the final label decisions.
Average precision provides a useful additional view when positives are rare.
Accuracy is not a primary metric because negative targets dominate.

## How similar shelf labels are handled

The primary task retains the original 15 labels for continuity with the group's
analysis. Related names such as `historical` and `historical-romance` do not
necessarily mean the same thing.

We separately study a 12-label target by merging these pairs:

| Original shelves | Merged label |
| --- | --- |
| historical, historical-romance | historical |
| contemporary, contemporary-romance | contemporary |
| paranormal, paranormal-romance | paranormal |

**Construct the original binary targets first, then combine each pair using
logical OR.** We do not add shelf counts: the same reader could use both shelves,
so adding votes would introduce an extra overlap assumption. OR also avoids
creating a positive simply because two individually sub-threshold counts add
up to 10%.

The merged experiment uses the selected architecture, fits a new classifier on
training data, and tunes new thresholds on validation. Its test micro-F1 is
{merged.micro_f1:.4f} and macro-F1 is {merged.macro_f1:.4f}.
These scores belong to a broader target and cannot be presented as a direct
gain on the original 15-label task. Training-label overlap statistics are in
`reports/full/alias_analysis.csv`.

## Code walkthrough

The [executed notebook](../modeling.ipynb) and [HTML export](../modeling.html)
display the actual functions and training loop, with explanatory text and
recorded outputs. The expensive training was run through `python -m modeling.run`;
the explanatory notebook reads those results rather than retraining eight models
whenever it is opened.

| Code | What to explain |
| --- | --- |
| `core.py`: `read_books`, `select_labels`, `make_targets` | Eligibility, training-only label selection, and binary targets |
| `core.py`: `split_books` | Keeping editions together and preserving the original test set |
| `core.py`: `make_features` | Sparse TF-IDF, word/character blocks, and fixed feature weights |
| `run.py`: `classifier`, `fit_checked` | OvR classifiers, L2 regularization, class weights, convergence checks, and controlled parallelism |
| `core.py`: `best_f1_threshold`, `tune_thresholds` | Exact threshold search, tied scores, and validation-only tuning |
| `run.py`: `main` | Candidate comparison, freezing selection, and final test evaluation |
| `core.py`: `merge_targets` | OR-based merging as a separate target |
| `predict.py`: `predict` | Applying the saved vectorizer, classifier, and thresholds to a new description |

Positive feature weights show associations that raise a shelf's score. They
are not causal explanations. Inspect `reports/full/top_features.csv` for examples.
The notebook includes a synthetic description for an inference demonstration.
`error_review_queue.csv` supplies candidate false-positive and false-negative
cases for human reading; it does not contain human-verified explanations.

## Limitations and validity checks

Although work groups do not overlap, the test set contains
{audit['exact_description_overlap']['test_books_with_text_in_train_or_validation']}
books whose lowercased, whitespace-normalized descriptions also occur in training
or validation. A diagnostic excluding these books retains
{unseen['test_books']:,} test examples and gives selected-model macro-F1
{unseen['selected']['macro_f1']:.4f}. This is a diagnostic subset, not a benchmark
retrained with all duplicate descriptions grouped together. Near duplicates
have not been completely removed.

Other limitations are:

- Shelf votes are noisy proxy labels, and the 10% / 10-vote rule has not been
  validated against human genre annotations.
- Authors and series can occur across partitions. The test does not establish
  generalization to new authors, future years, or other platforms.
- Bag-of-ngrams features have limited semantic understanding, and OvR does not
  explicitly model label relationships.
- Per-label thresholds may overfit smaller validation labels. Results come from
  one grouped partition and a finite candidate grid.

## Suggested four-minute presentation

| Time | Content and code to show |
| --- | --- |
| 0:00-0:40 | Description input, 15 binary outputs, and one vote-share labeling example |
| 0:40-1:35 | TF-IDF, independent LR classifiers, class imbalance, and regularization; show `make_features` and `classifier` |
| 1:35-2:30 | Default versus tuned thresholds; show `best_f1_threshold` and explain the validation partition |
| 2:30-3:20 | Validation comparison and selected model; explain the runtime and memory trade-offs |
| 3:20-4:00 | Separate label-merging experiment, one result table, and the main validity limitations |

Before presenting, run a prediction, inspect the leading feature weights for one
shelf, and read an error example. Explain the choices in your own words and keep
the claims consistent with the code and recorded results.

## Files and reproduction

| File or directory, relative to the repository root | Purpose |
| --- | --- |
| `modeling.ipynb`, `modeling.html` | Executed English modeling section with embedded outputs and figures |
| `modeling/core.py` | Data, targets, splitting, features, thresholds, and metrics |
| `modeling/run.py` | Full experiment, selection freeze, merged target, and bootstrap |
| `modeling/predict.py` | Prediction using a saved fitted model |
| `artifacts/full/selected_model.joblib` | Local selected classifier, vectorizer, labels, and thresholds |
| `artifacts/full/test_predictions.npz` | Test predictions and explicit book/label ordering |
| `reports/full/` | CSV tables, JSON decisions, split IDs, and PNG figures |
| `reports/full/verification.json` | Independent checks of exported outputs, split isolation, and metrics |
| `modeling/HANDOFF.md` | Instructions for merging with the shared workbook |

Use Python 3.11+ and run these commands from the repository root:

```bash
python -m pip install -r requirements-modeling.txt
python -m scripts.download_romance
python -m unittest discover -s tests -v
python -m modeling.run --jobs 4
python -m scripts.build_modeling_report
python -m scripts.verify_modeling
python -m modeling.predict "A young woman falls in love with a vampire."
```

The recorded full experiment took approximately {manifest['wall_seconds']/60:.1f}
minutes. On the original machine, `/home/cauthor/anaconda3/bin/python3` had the
dependencies; `/usr/bin/python3` did not. Other machines should use their own
configured Python environment. Reduce `--jobs` on machines with less available
RAM. Read [README.md](../README.md) for the setup details.

The original `code.py` is unchanged. Its name shadows Python's standard-library
`code` module and can interfere with IPython startup. The report builder handles
this collision; for interactive use, launch `PYTHONSAFEPATH=1 jupyter lab`.

Raw data and fitted artifacts are excluded from Git. The HTML and saved notebook
outputs can be read without retraining. Executing the notebook or prediction CLI
requires the local fitted artifacts, obtained by running the pipeline or sharing
those artifacts separately. Only load trusted local joblib files.

After integration, execute the combined group notebook from a fresh kernel and
export the final `workbook.html`. The group must also record the presentation
and supply its actual video URL.

## References

- [Official Goodreads dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html): dataset context and academic-use conditions.
- [scikit-learn text feature extraction](https://scikit-learn.org/1.6/modules/feature_extraction.html#text-feature-extraction).
- [scikit-learn logistic regression](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression).
- [scikit-learn support vector machines](https://scikit-learn.org/1.6/modules/svm.html#classification).
- [scikit-learn threshold tuning](https://scikit-learn.org/1.6/modules/classification_threshold.html).
- Wan and McAuley (2018), [Item Recommendation on Monotonic Behavior Chains](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf).
- Wan et al. (2019), [Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248/).

The last two papers provide dataset context. They study user-behavior-chain
recommendation and review spoiler detection, respectively. Their task scores
are not directly comparable with our description-to-shelf F1.
'''
    (ROOT/'modeling/MODELING_GUIDE_EN.md').write_text(english,encoding='utf-8')
    handoff=f'''# Modeling handoff

The executed `modeling.ipynb` and self-contained `modeling.html` are the modeling
contribution to the shared notebook. They are not the group's complete submission.
The assignment still needs the other members' task/EDA/evaluation/related-work
sections, an approximately 20-minute recorded presentation and its real video URL,
and the later peer-grading response. No fake video link or peer-grading file is supplied.

## Final choices

- Main task: original 15 shelves, original exclusion list and 10% / 10-vote rule.
- Label vocabulary: training data only; final order is in `reports/full/selection.json`.
- Split by work_id: train {audit['split_sizes']['train']:,}, validation {audit['split_sizes']['validation']:,},
  test {audit['split_sizes']['test']:,}. Outer seed 0 preserves the group's test IDs.
- Selected model: `{selection['selected_candidate']}`; selection by validation macro-F1.
- Thresholds: per-shelf F1 optima on validation only. No train+validation refit.
- Secondary task: 12 labels, OR merge of three alias pairs. Report separately.

## Integration contract

The cleanest integration is to adopt these split IDs for all group models.
`reports/full/splits.csv.gz` maps book_id/work_id to split in original eligible-row order.
`reports/full/label_support.csv` gives the names and supports. Always align labels
by name, not by an assumed index from the older global top-K list.
`artifacts/full/test_predictions.npz` arrays use the explicit `book_ids` order
stored inside that NPZ. Do not assume it is sorted or the same order as the split CSV.

For an evaluation-only handoff, use `y_true`, `scores`, `predictions`, `thresholds`,
`labels`, `book_ids`, and `work_ids` from that NPZ. `scores` are decision scores,
not calibrated probabilities. Compute AUC/AP from scores and F1 from predictions.
Keep zero-target-label books. Default- and tuned-threshold evaluations of the
same fitted model have identical AUC/AP.

`test_results.csv` contains only reference LR and the validation-selected model,
their predeclared default/global/per-label policies, prior and keyword baselines,
and the separate merged task. Other candidates are compared on validation in
`validation_search.csv`. Never use the test table to choose a different candidate.
The global threshold optimizes validation micro-F1; the per-label thresholds
optimize validation macro-F1. These are distinct operating goals.

`per_label_*.csv`, `top_features.csv`, `alias_analysis.csv`, `bootstrap.json`, and
`figures/` support evaluation and slides. `error_review_queue.csv` supplies 3
high-confidence FP/FN IDs per shelf for human review; it contains no human analysis.
Descriptions can be located in the local dataset or `artifacts/full/prepared.joblib`.

## Validity and interpretation

This run reserves validation, so the original plan's two-way-split scores are not
a matched comparison. Compare rows of the new test table with each other.
Raw shelf targets are noisy proxies, not human verified genre truth. The fixed
exclusion list is inherited rather than a complete ontology.
There are {audit['exact_description_overlap']['test_books_with_text_in_train_or_validation']} test books whose exact normalized description
appears in train/validation; a diagnostic excluding these is in
`unseen_description_sensitivity.json`. No claim of complete near-duplicate,
author, series, temporal or out-of-domain separation is made.
Bootstrap intervals resample test work groups and condition on the fitted models.

## Reproduce and export

Run the commands in README using the dependency-equipped Python interpreter.
`python -m modeling.run` retrains on the full local data; the report builder runs
the explanatory notebook from a fresh kernel using those recorded results.
It does not silently train 8 models again. All outputs and images are embedded
in HTML, whose first bytes are `<!DOCTYPE html>`.
The unchanged upstream `code.py` shadows Python's standard-library module.
The builder preloads the standard library and sets `PYTHONSAFEPATH=1` for its
fresh kernel. Use Python 3.11+ and `PYTHONSAFEPATH=1 jupyter lab` for interactive
notebook work in this repository.
After copying the section into the shared workbook, execute that combined
notebook from a fresh kernel and export the group's final **workbook.html**.
Do not rename a notebook to .html. Record and validate the real video separately.

Large data and fitted artifacts are intentionally ignored by Git. Preserve the
original `code.py`. The local clone's upstream base commit is
`{manifest['upstream_commit']}`.
'''
    (ROOT/'modeling/HANDOFF.md').write_text(handoff,encoding='utf-8')


def notebook(audit, selection):
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
    md('''# Predicting Goodreads romance shelves from descriptions

**Modeling contribution for CSE 258R Assignment 2.** This section defines the
learning problem, compares sparse linear models, tunes thresholds without test
leakage, and studies a separate coarser-label target. All tables and figures below
are actual full-data outputs. The expensive experiment was executed with
`python -m modeling.run`; this notebook reads its recorded results and runs from
a fresh kernel. The group will merge this with the other four assignment sections.
''')
    code('''from pathlib import Path
import inspect, json, sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Code, Markdown

ROOT = Path.cwd()
if not (ROOT / "reports/full").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from modeling import core, run
from modeling.predict import predict
R = ROOT / "reports/full"
def read_json(name):
    return json.loads((R / name).read_text())
audit = read_json("data_audit.json")
selection = read_json("selection.json")
manifest = read_json("run_manifest.json")
assert audit["raw_books"] == 335449 and not audit["partial_data"]
assert selection["test_used_for_selection"] is False
assert manifest["selection_frozen_before_test"] is True
print("Full romance dataset:", audit["raw_books"], "records")
print("Experiment runtime:", round(manifest["wall_seconds"]/60, 2), "minutes")
print("Selected using validation only:", selection["selected_candidate"])
display(Image(filename=str(R / "figures/modeling_pipeline.png")))
''')
    md(r'''## Learning problem and supervision

Input $d_i$ is **description text only**. Output $y_i\in\{0,1\}^{15}$ indicates
which selected shelf labels meet the group's vote-share rule. A book may have
several labels or none. This supports content tagging for discovery, rather than
personalized user-item recommendation. We have no per-user interactions in this
metadata file, so collaborative LFM/BPR is not a direct fit to this task.

After the fixed exclusions in the original repository, let $v_i$ be the sum of
remaining shelf counts. Keep nonempty descriptions with $v_i\geq10$, and set
$y_{ij}=\mathbf{1}[c_{ij}/v_i\geq0.10]$. Select the 15 most frequent positive
shelves **from the training partition only**. Zero labels mean no selected shelf
passes the heuristic, not that a human verified the absence of those genres.
The inherited exclusion list still leaves some non-content shelves in the
denominator; these are weak, noisy labels rather than a curated ontology.
We retain `manga` as an observed shelf even though it describes a medium rather
than a romance subgenre; the target is reader shelf usage.
''')
    code('''display(pd.Series({key: audit[key] for key in [
    "raw_books", "nonempty_descriptions", "eligible_books",
    "mean_labels_per_book", "zero_label_fraction", "missing_work_id_fallbacks"]}, name="value"))
support = pd.read_csv(R / "label_support.csv")
display(support)
display(Markdown("Excluded shelves: `" + "`, `".join(sorted(core.EXCLUDE)) + "`"))
display(Code(inspect.getsource(core.select_labels) + "\\n" + inspect.getsource(core.make_targets), language="python"))
''')
    md('''## Work-disjoint train, validation and test

Editions of the same work stay together. The original outer `GroupShuffleSplit`
uses seed 0 and a 20% test fraction by work group. A second grouped split with
seed 1 reserves 20% of the development works for validation. Because work sizes
differ, row fractions are approximate. Vectorizers and classifiers fit on train;
model choice and thresholds use validation; test is scored after selection is
written to disk. A missing work ID would fall back to a per-book key; there were
none in this full run.

We keep the model fitted on train after threshold selection. Refitting on
train+validation could shift its score scale and invalidate these thresholds.
The earlier plan's two-way split used more training books, so compare the
matched ablations below rather than claiming a direct gain over that old run.
''')
    code('''display(pd.DataFrame({"books":audit["split_sizes"], "works":audit["split_work_counts"]}))
print("Work-group overlap:", audit["group_overlap"])
display(Code(inspect.getsource(core.split_books), language="python"))
''')
    md(r'''## Text representation

Word TF-IDF uses unigrams and bigrams, 50,000 features, `min_df=5`, English
stop-word removal and sublinear term frequency. A component is proportional to
$(1+\log tf)\,[\log((1+n)/(1+df))+1]$, followed by L2 normalization. Vocabulary
and IDF are fitted on training descriptions only.

The richer representation concatenates word features with up to 60,000
`char_wb` 3–5 character n-grams. Each block is normalized separately; fixed
weights are 1.0 for words and 0.5 for characters. Character fragments can help
with morphology, spellings and non-English text, at extra memory and runtime
cost. Neither representation sees titles, authors, ratings, or shelf counts.
''')
    code('display(Code(inspect.getsource(core.make_features), language="python"))')
    md(r'''## Algorithms and objectives

One-vs-rest (OvR) fits one independent binary classifier per shelf. For label
$j$, its linear score is $s_{ij}=w_j^Tx_i+b_j$ and $z_{ij}=2y_{ij}-1$.
With `intercept_scaling=1`, liblinear also regularizes the synthetic intercept
weight; write $\widetilde w=(w,b)$.

**Logistic regression** minimizes
$\tfrac12\|\widetilde w_j\|_2^2+C\sum_i a_{ij}\log(1+\exp(-z_{ij}s_{ij}))$.
**LinearSVC** minimizes the analogous L2-regularized **squared hinge** objective
$\tfrac12\|\widetilde w_j\|_2^2+C\sum_i a_{ij}\max(0,1-z_{ij}s_{ij})^2$.
Higher C weakens regularization relative to data fit. Balanced class weights are
$a_{ij}=n/(2n_{j,y_{ij}})$, computed on training data for each OvR classifier.

LR is easy to interpret and supports sparse text efficiently. Its weighted
sigmoid scores should not be treated as calibrated real-world probabilities.
Linear SVM also scales to sparse features and emphasizes separation, but returns
margins rather than probabilities. OvR does not explicitly model label
dependencies; bag-of-ngrams cannot reliably capture long-range meaning.
Training and prediction operate on sparse nonzeros rather than dense document
vectors; prediction costs approximately $O(K\,nnz(X))$. The word+character
model needs more nonzeros and K weight vectors of length 110,000 rather than
50,000. A nonlinear kernel matrix is unnecessary here.
''')
    code('''display(pd.DataFrame(run.CANDIDATES))
display(Code(inspect.getsource(run.classifier) + "\\n" + inspect.getsource(run.fit_checked), language="python"))
''')
    md(r'''## Thresholds turn scores into labels

For a fitted classifier, predict $\widehat y_{ij}=\mathbf{1}[s_{ij}\geq t_j]$.
We compare a default threshold 0, one global threshold maximizing validation
micro-F1, and one threshold per label maximizing its validation F1. Independent
per-label F1 optimization corresponds to maximizing the mean (macro-F1).
It need not maximize micro-F1.

The implementation sorts each label's validation scores, updates cumulative TP,
and evaluates F1 only at the end of a complete equal-score block. This costs
$O(n\log n)$ per label. Ties keep the highest threshold; a label with no
validation positives predicts none. We consistently use `>=`, including at
inference. Threshold changes do not change AUC/AP because the scores and their
rankings stay fixed. Thresholds are not tuned on test.
''')
    code('''display(Code(inspect.getsource(core.best_f1_threshold) + "\\n" + inspect.getsource(core.tune_thresholds), language="python"))
demo_y = np.array([0, 1, 1, 0, 0])
demo_scores = np.array([-1.2, .4, .4, .1, -.3])
demo_t = core.best_f1_threshold(demo_y, demo_scores)
print("Synthetic tied-score example: threshold =", demo_t)
print("Predictions:", (demo_scores >= demo_t).astype(int))
''')
    md('''## Validation comparison and frozen model selection

Eight candidates and three operating policies were declared before test
evaluation. Select the per-label-threshold candidate with the highest validation
macro-F1; break ties by micro-F1 and then name. This gives each shelf equal
importance in selection. The global threshold has the different objective of
micro-F1. The full training sequence is in `modeling/run.py`; the table records
actual measured runtime, including validation scoring in `train_seconds`.
Feature construction time is shared by candidates of the same representation.
''')
    code('''search = pd.read_csv(R / "validation_search.csv")
display(search[["candidate","policy","micro_f1","macro_f1","macro_auc", "train_seconds","feature_seconds","n_features"]].round(4))
display(Image(filename=str(R / "figures/validation_search.png")))
display(pd.Series(selection))
''')
    md(r'''## Held-out test ablations

Micro-F1 pools TP/FP/FN over book-label pairs, so frequent shelves matter more.
Macro-F1 averages F1 equally across shelves and is our selection objective.
ROC-AUC measures ranking of positives above negatives, while average precision
(AP) adds a useful view when positives are rare. High AUC alone does not imply
the chosen labels are accurate. Ordinary accuracy is uninformative with many
negative targets.

The prior baseline returns training prevalences and thresholds them at 0.5.
The literal keyword baseline detects the shelf name in the description after
simple punctuation/space normalization. These have no fitted text model.
Only the reference LR and the validation-selected candidate are scored on the
original held-out test, under their predeclared threshold policies. We do not
switch models based on this table.
The global policy can produce higher test micro-F1 than the per-label policy;
the primary configuration follows the validation macro-F1 criterion declared
before test evaluation, rather than selecting the best-looking test row.
''')
    code('''test = pd.read_csv(R / "test_results.csv")
display(test[test.task == "original_15"][["candidate","policy","micro_f1","macro_f1", "macro_auc","macro_ap","micro_precision","micro_recall"]].round(4))
display(Image(filename=str(R / "figures/test_ablation.png")))
per_shelf = pd.read_csv(R / f"per_label_{selection['selected_candidate']}.csv")
display(per_shelf.round(4))
display(Image(filename=str(R / "figures/per_shelf_f1.png")))
''')
    md('''## Uncertainty and repeated-text diagnostic

The paired bootstrap resamples held-out **work groups**, retaining their
editions, using the same resampled groups for both models. The 95% percentile
intervals use 300 replicates and condition on the fitted models. They do not
include uncertainty from retraining or model selection.

Work grouping does not remove all repeated descriptions across different IDs.
We audit exact hashes of lowercased whitespace-normalized descriptions, and
report a test subset excluding texts found in either train or validation. This
is a diagnostic, not a new deduplicated retraining experiment. Near duplicates,
shared authors and shared series can remain. Random work splits do not establish
temporal or new-author generalization.
''')
    code('''display(pd.json_normalize(read_json("bootstrap.json"), sep=" / ").T)
display(pd.Series(audit["exact_description_overlap"]))
unseen = read_json("unseen_description_sensitivity.json")
print("Unseen exact-text subset:", unseen["test_books"], "books")
display(pd.DataFrame({k:unseen[k] for k in ["selected","reference_default","reference_tuned"]}).T.round(4))
''')
    md('''## Shelf alias decision and a separate target

Retain all 15 shelves for the primary group task. Historical and historical
romance are related but not identical concepts, as are the contemporary and
paranormal pairs. As sensitivity analysis, coarsen these three pairs to 12
labels by OR-ing their **original binary targets**. Do not sum vote counts:
the same reader could use both shelves. This OR rule does not create positives
merely because two individually sub-threshold counts add up to 10%.

Use the selected architecture, fit a new 12-output model on train and tune new
thresholds on validation. The task and its target prevalences have changed;
an F1 difference is not a direct model improvement on the original task.
''')
    code('''display(Code(inspect.getsource(core.merge_targets), language="python"))
display(pd.read_csv(R / "alias_analysis.csv").round(4))
display(test[test.task != "original_15"][["task","candidate","micro_f1","macro_f1","macro_auc","macro_ap"]].round(4))
display(pd.Series(read_json("merge_decision.json")))
''')
    md('''## Interpretation and a prediction example

Positive weights identify features that increase a shelf score. They are
associations in noisy Goodreads data, not causal explanations. In the richer
representation, `word__` and `char__` prefixes identify feature blocks.
The error table queues held-out books for manual reading by the evaluation
member; error reasons have not been automatically declared as human findings.
The following description is synthetic and demonstrates inference only.
''')
    code('''terms = pd.read_csv(R / "top_features.csv")
display(terms[(terms.direction == "positive") & (terms["rank"] <= 3)])
example = "A young woman falls in love with a vampire while a supernatural war threatens their town."
print("Synthetic description:", example)
example_scores = pd.DataFrame(predict(example, ROOT / "artifacts/full/selected_model.joblib"))
display(example_scores.sort_values("score", ascending=False).round(4))
display(pd.read_csv(R / "error_review_queue.csv").head(10).round(4))
''')
    md('''## Appendix: the actual training and selection loop

The following is the experiment driver's actual source, including feature fits,
candidate fitting, validation threshold search and the selection freeze. It was
executed by the full-data CLI run; this display is a code walkthrough and does
not silently launch a second training run. The next part of the driver performs
the separate merged task and only then scores test data.
''')
    code('''training_source = inspect.getsource(run.main)
start = training_source.index("    validation_rows, bundles")
end = training_source.index("    # Separate coarser-label")
display(Code(training_source[start:end], language="python"))
''')
    md('''## Related sources and reproducibility

The [official Goodreads dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html)
describes the late-2017 public-shelf collection and the academic-use conditions.
Wan and McAuley (2018), [Item Recommendation on Monotonic Behavior Chains](https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf),
study recommendation across ordered user behaviors. Wan et al. (2019),
[Fine-Grained Spoiler Detection from Large-Scale Review Corpora](https://aclanthology.org/P19-1248/),
study sentence-level spoiler detection. These provide dataset context; their
task scores are not comparable to our description-to-shelf F1.

Implementation references: scikit-learn 1.6 documentation for
[text features](https://scikit-learn.org/1.6/modules/feature_extraction.html#text-feature-extraction),
[logistic regression](https://scikit-learn.org/1.6/modules/linear_model.html#logistic-regression),
[LinearSVC](https://scikit-learn.org/1.6/modules/svm.html#classification),
[threshold tuning](https://scikit-learn.org/1.6/modules/classification_threshold.html), and
[metrics](https://scikit-learn.org/1.6/modules/model_evaluation.html).

Reproduce from the repository root in the documented Python environment:

```bash
python -m scripts.download_romance
python -m unittest discover -s tests -v
python -m modeling.run --jobs 4
python -m scripts.build_modeling_report
```

Raw data and fitted artifacts stay local and are excluded from Git. The original
`code.py` remains intact. The notebook and HTML embed all their outputs; the
grader does not need to execute them. The shared group workbook still needs the
other members' sections and must be rerun and exported after integration.
''')
    code('''print("Dataset SHA-256:", audit["dataset_sha256"])
print("Dataset bytes:", audit["dataset_bytes"])
print("Upstream base commit:", manifest["upstream_commit"])
display(pd.Series({k:manifest[k] for k in ["python","numpy","scipy","pandas","sklearn","jobs","inner_threads","split_seeds","wall_seconds"]}))
''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}})
    client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbf.validate(nb)
    nbf.write(nb,ROOT/'modeling.ipynb')
    body,_=HTMLExporter(template_name='lab',embed_images=True).from_notebook_node(nb)
    body=body.lstrip()
    assert body.startswith('<!DOCTYPE html>')
    (ROOT/'modeling.html').write_text(body,encoding='utf-8')
    print(f'Wrote {len(cells)} executed cells and {len(body):,} HTML characters.')


def main():
    audit,selection,manifest=map(read_json,['data_audit.json','selection.json','run_manifest.json'])
    if audit['partial_data']:
        raise ValueError('Final report requires a full-data run')
    search=pd.read_csv(REPORT/'validation_search.csv')
    test=pd.read_csv(REPORT/'test_results.csv')
    diagnostics=[]
    for name in search.candidate.unique():
        bundle=joblib.load(ROOT/'artifacts/full'/f'{name}.joblib')
        estimators=bundle['classifier'].estimators_
        diagnostics.append({
            'candidate':name, 'classifier':type(estimators[0]).__name__,
            'max_solver_iterations':max(int(np.max(e.n_iter_)) for e in estimators),
            'median_solver_iterations':float(np.median([np.max(e.n_iter_) for e in estimators])),
            'coefficient_bytes':sum(e.coef_.nbytes for e in estimators),
            'convergence_warning':False})
    pd.DataFrame(diagnostics).to_csv(REPORT/'optimization_diagnostics.csv',index=False)
    figures(selection,search,test)
    guides(audit,selection,test,manifest,read_json('bootstrap.json'),
           read_json('unseen_description_sensitivity.json'))
    notebook(audit,selection)


if __name__=='__main__':
    main()
