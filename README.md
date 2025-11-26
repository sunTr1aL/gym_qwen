# Speculative TD-MPC2 with Corrector

This repo extends the original **TD-MPC2** implementation (in `tdmpc2/`) with:

- **Speculative execution** over multi-step TD-MPC2 plans
- A learned **corrector** network that distills TD-MPC2’s replanning behavior
- Support for **multiple corrector architectures** (gated two-tower MLP, temporal Transformer)
- Scripts to **collect data**, **train the corrector**, and **evaluate** long-horizon performance

The core idea:

> Let TD-MPC2 plan a short horizon (e.g. 3 steps), then execute those steps directly.  
> When the real state deviates from the predicted state, use a small corrector network to approximate  
> *“what TD-MPC2 would have done if it replanned from the real state”* — without paying the full MPC cost.

This is intended to **accelerate inference** and explore how far we can safely extend TD-MPC2’s control horizon (e.g. from 3 to 6 steps) without sacrificing too much performance.

---

## High-Level Design

### 1. Speculative Execution

In vanilla TD-MPC2, at each timestep:

1. Encode observation: \(s_t \to z_t\)
2. Run MPC in latent space with horizon \(H\) (e.g. 3)
3. Execute only the **first** action \(a_t\), then replan at the next step

In this extension:

1. At time \(t\), TD-MPC2 plans a 3-step sequence:
   - Actions: \((a_t^{\text{plan}}, a_{t+1}^{\text{plan}}, a_{t+2}^{\text{plan}})\)
   - Predicted latent states: \((z_t, z_{t+1}^{\text{pred}}, z_{t+2}^{\text{pred}}, z_{t+3}^{\text{pred}})\)
2. Instead of replanning at every step, we **execute the plan directly** over several env steps
3. At each step \(t+k\), we compare:
   - Real encoded latent \(z_{t+k}^{\text{real}}\) from the environment
   - Predicted latent \(z_{t+k}^{\text{pred}}\) from the plan
4. If mismatch is small, we **correct** the planned action via a small learned network and execute it
5. If mismatch is too large, we **fall back to a full TD-MPC2 replan** from the real state

This makes TD-MPC2 act more like a short-horizon **open-loop controller with learned feedback**, instead of replanning at every step.

---

### 2. Corrector Networks

The corrector is trained to approximate the *teacher* TD-MPC2:

- Teacher action at real state \(s_{t+1}^{\text{real}}\):  
  \(a_{t+1}^{\text{teach}} = \text{TD-MPC2}(s_{t+1}^{\text{real}})\)
- Planned action for step \(t+1\):  
  \(a_{t+1}^{\text{plan}}\) from the 3-step plan at time \(t\)
- Predicted vs real latent mismatch:  
  \(z_{t+1}^{\text{pred}}\) vs \(z_{t+1}^{\text{real}}\)

The corrector takes these (plus optional history) and outputs:

\[
a_{t+1}^{\text{corr}} = a_{t+1}^{\text{plan}} + \Delta a_{t+1} \approx a_{t+1}^{\text{teach}}
\]

Two architectures are supported:

1. **TwoTowerCorrector** (gated two-tower MLP)
   - Separate towers for real latent, predicted latent, and their difference
   - Gating mechanism decides how much to trust the planned action versus a learned update

2. **TemporalTransformerCorrector**
   - Uses a short history of mismatch features
   - Encodes a sequence of past \((z^{\text{real}}, z^{\text{pred}}, \Delta z, a^{\text{plan}})\)
   - Applies a small Transformer encoder to capture systematic model biases / drift over time

Both output a **residual correction** on top of the planned action.

---

### 3. Distillation Training

We train the corrector in a **distillation** style: it learns to mimic what TD-MPC2 would output if it replanned from the real state.

For each training sample:

- Input:
  - \(z_{t+1}^{\text{real}}\): latent from real next state
  - \(z_{t+1}^{\text{pred}}\): latent from predicted next state in the plan
  - \(a_{t+1}^{\text{plan}}\): planned action at t+1
  - Optional `history_feats`: short history of mismatch features for temporal correctors
- Target:
  - \(a_{t+1}^{\text{teach}} = \text{TD-MPC2}(s_{t+1}^{\text{real}})\): teacher action from a fresh plan

Loss:

- Distillation loss:
  \[
  L_{\text{mse}} = \|a_{t+1}^{\text{corr}} - a_{t+1}^{\text{teach}}\|^2
  \]
- Residual regularization:
  \[
  L_{\text{reg}} = \lambda \|\Delta a_{t+1}\|^2
  \]
- Total:
  \[
  L = L_{\text{mse}} + L_{\text{reg}}
  \]

We can also add a **multi-step rollout loss** by unrolling the corrector over several steps (e.g. 6) and matching the full sequence of teacher actions.

---

### 4. Long-Horizon Evaluation

To test whether the corrector “actually understands” TD-MPC2 and can be reused over time, we evaluate:

1. **Baseline TD-MPC2**  
   - Replan every step
   - Executes only the first action of each MPC plan

2. **Naive 3-step open-loop**  
   - Plan 3 steps, execute all 3 with **no correction**
   - Replan every 3rd step

3. **3-step + Corrector**  
   - Plan 3 steps at time t
   - At t+1, t+2:
     - Compare predicted vs real latent
     - Use corrector to adjust planned action
   - Replan at t+3

4. **6-step + Corrector** (stress test)  
   - Start from a 3-step plan
   - Use the world model + corrector to **extend effective execution to 6 steps**
   - Replan at t+6

We compare:

- Episode returns (mean, median, worst 5%)
- Failure/early-termination rate
- Frequency of fallbacks to full TD-MPC2 replanning
- Magnitude of corrections (‖Δa‖)
- Performance as a function of step index within the plan (t+1 … t+6)

---

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

---

## Setup

Assuming you’re in the root of the `gym_qwen` repo:

```bash
# Create environment, install deps (example)
conda create -n tdmpc2_spec python=3.10
conda activate tdmpc2_spec

pip install -r requirements.txt
# plus mujoco/dm_control/gym dependencies as required by the original repo
