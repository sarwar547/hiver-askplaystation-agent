from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

GOLDEN = DATA / 'playstation_golden_set_labeled.csv'
PAIRS = DATA / 'playstation_customer_support_pairs.csv'

# Keyword weak labels are used only to create a scalable training set.
KEYWORDS = {
    'account_access':['login','log in','sign in','password','account','access','locked out','username','email','dob'],
    'account_security':['hack','hacked','security','stolen','unauthorized','ban','banned','suspended','2fa'],
    'payments_billing':['charged','charge','payment','billing','credit card','debit card','transaction','funds'],
    'refunds':['refund','refunds','money back','return','cancel purchase'],
    'ps_plus_subscription':['ps plus','playstation plus','ps+','subscription','renewal','renew','monthly games'],
    'game_purchase_store':['buy','bought','purchase','purchased','store','game sharing','digital game','price','sale'],
    'download_installation':['download','downloading','install','installation','installing','corrupt','corrupted','update'],
    'console_hardware':['ps4','ps5','console','controller','disc','eject','power','overheat','button','safe mode'],
    'network_psn':['network','internet','wifi','wi-fi','connection','connect','psn','nw-','dns','online'],
    'codes_dlc_content':['code','voucher','redeem','dlc','add-on','addon','points','entitlement'],
    'content_availability':['available','availability','release','added','catalogue','catalog','offerings'],
}

def weak_label(text):
    t = str(text).lower()
    scores = {k: sum(v in t for v in kws) for k,kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'other'

pairs = pd.read_csv(PAIRS, dtype=str).fillna('')
pairs['weak_intent'] = pairs['customer_message'].map(weak_label)

# Remove the golden IDs from retrieval to prevent evaluation leakage.
golden = pd.read_csv(GOLDEN, dtype=str).fillna('')
golden_ids = set(golden.get('customer_tweet_id', pd.Series(dtype=str)).astype(str))
retrieval = pairs[~pairs['customer_tweet_id'].astype(str).isin(golden_ids)].copy()
if retrieval.empty:
    retrieval = pairs.copy()

# Train on the historical pairs with weak labels. Golden set remains evaluation-only.
model = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=60000, sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
model.fit(pairs['customer_message'], pairs['weak_intent'])
joblib.dump(model, ART / 'intent_model.joblib')

vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=80000, sublinear_tf=True)
X = vec.fit_transform(retrieval['customer_message'])
joblib.dump(vec, ART / 'retrieval_vectorizer.joblib')
joblib.dump(X, ART / 'retrieval_matrix.joblib')
retrieval[['customer_tweet_id','customer_message','support_response']].reset_index(drop=True).to_pickle(ART / 'retrieval_rows.pkl')

print(f'Training pairs: {len(pairs):,}')
print(f'Retrieval rows (golden IDs removed): {len(retrieval):,}')
print('Artifacts written to artifacts/')
