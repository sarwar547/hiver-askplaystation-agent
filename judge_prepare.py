from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
pred=ROOT/'reports/golden_predictions.csv'
out=ROOT/'reports/judge_human_sample.csv'

df=pd.read_csv(pred,dtype=str).fillna('')
# Fixed sample for reproducibility. Human scores the same rows later.
sample=df.sample(n=min(50,len(df)),random_state=123).copy()
for c in ['human_groundedness','human_correctness','human_actionability','human_safety','human_no_hallucination']:
    sample[c]=''
sample.to_csv(out,index=False)
print(f'Created {len(sample)}-example human judge sample: {out}')
print('Score each dimension 0, 1, or 2: 0=bad, 1=partial, 2=good.')
