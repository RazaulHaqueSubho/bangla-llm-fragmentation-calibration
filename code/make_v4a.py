# make_v4a.py
import json
import re

with open('Bangla-LLM-Calibration-Real-Data-v4a.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Loaded v4a notebook with', len(nb['cells']), 'cells')
