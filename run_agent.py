from src.agent import PlaystationAgent

agent = PlaystationAgent('artifacts')
print('AskPlayStation support agent. Type quit to exit.')
while True:
    text = input('\nCustomer: ').strip()
    if text.lower() == 'quit':
        break
    out = agent.run(text)
    print('\nIntent:', out['intent'], f"({out['intent_confidence']})")
    print('Decision:', out['action'])
    print('Reason:', out['reason'])
    print('Draft reply:', out['reply'])
    print('\nEvidence:')
    for e in out['evidence']:
        print(f"  similarity={e['similarity']} | {e['customer_message']}")
        print(f"  historical response: {e['historical_response']}")
