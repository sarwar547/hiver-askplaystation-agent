from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parent
GOLDEN=ROOT/'data/playstation_golden_set_labeled.csv'

df=pd.read_csv(GOLDEN,dtype=str).fillna('')
df=df[df['intent'].str.strip()!=''].copy()

majority_intent=df['intent'].mode().iloc[0]
majority_action=df['expected_action'].mode().iloc[0]

print('TRIVIAL BASELINE')
print('majority intent:', majority_intent)
print('intent accuracy:', round(accuracy_score(df['intent'], [majority_intent]*len(df)),4))
print('intent macro F1:', round(f1_score(df['intent'], [majority_intent]*len(df), average='macro', zero_division=0),4))
print('action accuracy:', round(accuracy_score(df['expected_action'], [majority_action]*len(df)),4))

KEYWORDS={
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

def pred(t):
    t=t.lower()
    scores={k:sum(x in t for x in v) for k,v in KEYWORDS.items()}
    best=max(scores,key=scores.get)
    return best if scores[best]>0 else 'other'

p=df['customer_message'].map(pred)
print('\nSIMPLE KEYWORD BASELINE')
print('intent accuracy:',round(accuracy_score(df['intent'],p),4))
print('intent macro F1:',round(f1_score(df['intent'],p,average='macro',zero_division=0),4))
