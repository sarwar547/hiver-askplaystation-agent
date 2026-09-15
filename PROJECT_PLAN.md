# End-to-end build plan

1. Data extraction and conversation reconstruction — DONE
2. Golden set — DONE (200 examples)
3. Golden-set audit — run before final results
4. Weak-label scalable training + historical retrieval — `train.py`
5. Trivial/simple/proposed evaluation — `baselines.py`, `evaluate.py`
6. Human/LLM reply evaluation — `judge_prepare.py`, `judge.py`, `human_agreement.py`
7. Failure analysis from prediction CSV
8. Final 6-page report
9. GitHub cleanup + README reproducibility check
