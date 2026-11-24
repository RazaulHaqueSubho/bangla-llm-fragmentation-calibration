import json,re
nb=json.load(open('e2_run/Bangla-LLM-Calibration-E2-rescue-v4h.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code': continue
    src=''.join(c['source'])
    for n,l in enumerate(src.splitlines(),1):
        if re.search(r'/kaggle/input|bnmmlu|BnMMLU|load_dataset',l,re.I):
            print(f"cell {i} line {n}: {l.strip()[:150]}")
