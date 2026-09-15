from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT=Path(__file__).resolve().parent
p=ROOT/'reports/llm_judge_scores.csv'
df=pd.read_csv(p,dtype=str).fillna('')

dims=['groundedness','correctness','actionability','safety','no_hallucination']
for d in dims:
    h=pd.to_numeric(df['human_'+d],errors='coerce')
    l=pd.to_numeric(df['llm_'+d],errors='coerce')
    m=h.notna() & l.notna()
    if m.sum()<2:
        continue
    print(d, 'n=',int(m.sum()), 'weighted_kappa=',round(cohen_kappa_score(h[m],l[m],weights='quadratic'),3), 'exact_agreement=',round((h[m]==l[m]).mean(),3))
