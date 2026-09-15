from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.agent import PlaystationAgent

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

GOLDEN = DATA / 'playstation_golden_set_labeled.csv'

df = pd.read_csv(GOLDEN, dtype=str).fillna('')
df = df[df['intent'].str.strip() != ''].copy()
agent = PlaystationAgent(ROOT / 'artifacts')

pred_intents=[]
pred_actions=[]
rows=[]
for _, r in df.iterrows():
    out = agent.run(r['customer_message'])
    pred_intents.append(out['intent'])
    pred_actions.append(out['action'])
    rows.append({
        'customer_tweet_id':r.get('customer_tweet_id',''),
        'customer_message':r['customer_message'],
        'gold_intent':r['intent'],
        'pred_intent':out['intent'],
        'intent_confidence':out['intent_confidence'],
        'gold_action':r['expected_action'],
        'pred_action':out['action'],
        'escalation_reason':out['reason'],
        'reply':out['reply'],
        'top_evidence':out['evidence'][0]['historical_response'] if out['evidence'] else ''
    })

result = pd.DataFrame(rows)
result.to_csv(REPORTS / 'golden_predictions.csv', index=False)

metrics = {
    'n': len(df),
    'intent_accuracy': accuracy_score(df['intent'], pred_intents),
    'intent_macro_f1': f1_score(df['intent'], pred_intents, average='macro', zero_division=0),
    'action_accuracy': accuracy_score(df['expected_action'], pred_actions),
    'intent_report': classification_report(df['intent'], pred_intents, output_dict=True, zero_division=0),
    'confusion_matrix': confusion_matrix(df['intent'], pred_intents, labels=sorted(df['intent'].unique())).tolist(),
    'labels': sorted(df['intent'].unique())
}
with open(REPORTS / 'metrics.json','w') as f:
    json.dump(metrics,f,indent=2)

print(json.dumps({k:v for k,v in metrics.items() if k in ['n','intent_accuracy','intent_macro_f1','action_accuracy']}, indent=2))
print('\nPredictions saved to reports/golden_predictions.csv')
