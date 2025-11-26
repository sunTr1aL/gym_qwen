## Repository Layout (new pieces)

Key additions on top of the original `tdmpc2` code:

- `tdmpc2/agent.py`
  - Speculative execution logic
  - Plan buffers, mismatch computation, fallback to TD-MPC2
  - Integration with corrector

- `tdmpc2/corrector.py`
  - `BaseCorrector`
  - `TwoTowerCorrector`
  - `TemporalTransformerCorrector`

- `scripts/collect_corrector_data.py`
  - Run TD-MPC2 as teacher
  - Collect distillation tuples for the corrector

- `scripts/train_corrector.py`
  - Train corrector offline on the collected dataset

- `scripts/eval_corrector.py`
  - Evaluate baseline, naive 3-step, 3-step+corrector, 6-step+corrector
  - Compare corrector architectures
