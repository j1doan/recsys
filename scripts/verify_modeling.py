"""Cross-check recorded metrics, split isolation, model inference and export."""
import json
import subprocess
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from modeling.predict import predict


def main():
    root=Path(__file__).resolve().parents[1]
    report=root/'reports/full'
    selection=json.loads((report/'selection.json').read_text())
    audit=json.loads((report/'data_audit.json').read_text())
    p=np.load(root/'artifacts/full/test_predictions.npz',allow_pickle=False)
    assert p['labels'].tolist()==selection['labels']
    np.testing.assert_allclose(p['thresholds'],selection['thresholds'])
    np.testing.assert_array_equal(p['predictions'],p['scores']>=p['thresholds'])
    table=pd.read_csv(report/'test_results.csv')
    assert table.groupby(['candidate','task']).macro_auc.nunique().max()==1
    assert table.groupby(['candidate','task']).macro_ap.nunique().max()==1
    row=table[(table.candidate==selection['selected_candidate']) &
              (table.policy=='per_label') & (table.task=='original_15')].iloc[0]
    calculated={
        'micro_f1':f1_score(p['y_true'],p['predictions'],average='micro'),
        'macro_f1':f1_score(p['y_true'],p['predictions'],average='macro'),
        'macro_auc':roc_auc_score(p['y_true'],p['scores'],average='macro'),
        'macro_ap':average_precision_score(p['y_true'],p['scores'],average='macro')}
    for key,value in calculated.items():
        np.testing.assert_allclose(row[key],value,atol=1e-12)
    splits=pd.read_csv(report/'splits.csv.gz',dtype=str,keep_default_na=False)
    assert not splits.book_id.duplicated().any()
    assert splits.groupby('work_id').split.nunique().max()==1
    test_ids=set(splits.loc[splits.split=='test','book_id'])
    assert test_ids==set(p['book_ids']) and len(test_ids)==60410
    assert len(splits)==299789
    notebook=nbformat.read(root/'modeling.ipynb',as_version=4)
    nbformat.validate(notebook)
    code_cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type=='error' for c in code_cells for o in c.outputs)
    html=(root/'modeling.html').read_text()
    assert html.startswith('<!DOCTYPE html>')
    soup=BeautifulSoup(html,'html.parser')
    images=soup.find_all('img')
    assert len(images)>=3 and all(i.get('src','').startswith('data:image/') for i in images)
    example=predict('A young woman falls in love with a vampire.',
                    root/'artifacts/full/selected_model.joblib')
    assert len(example)==15 and all(np.isfinite(x['score']) for x in example)
    subprocess.run(['git','diff','--exit-code','--','code.py'],cwd=root,check=True,
                   stdout=subprocess.DEVNULL)
    result={'passed':True,'eligible_books':len(splits),'test_books':len(test_ids),
            'independent_metrics':calculated,'executed_notebook_cells':len(code_cells),
            'embedded_plot_images':len(images),'original_code_unchanged':True,
            'full_data':not audit['partial_data'], 'work_overlap':0}
    (report/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
