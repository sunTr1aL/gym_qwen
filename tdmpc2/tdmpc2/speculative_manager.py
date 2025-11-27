from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from tdmpc2.common import math


@dataclass
class SpeculativeBranch:
    """Container for a speculative trajectory branch.

    Attributes:
        z_sequence: Predicted latent states for the speculative rollout. The first
            element corresponds to the latent prediction for the next real
            observation (``z_{t+1}``).
        actions: Planned speculative actions starting at ``t+1``.
        cumulative_reward: Predicted return accumulated over the speculative depth.
        dyn_cache: Optional dynamics cache when transformer dynamics are enabled.
    """

    z_sequence: List[torch.Tensor]
    actions: List[torch.Tensor]
    cumulative_reward: torch.Tensor
    dyn_cache: Optional[object] = None


class SpeculativeManager:
    """Manage speculative latent rollouts between environment steps.

    The manager keeps a small beam of latent trajectories that are rolled out while
    the environment is executing the previous action. When the next observation
    arrives, it matches the real latent to the speculative predictions and reuses
    the best branch when possible.
    """

    def __init__(self, model, cfg, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.beam_width = int(getattr(cfg, "beam_width", 4))
        self.spec_depth = int(getattr(cfg, "spec_depth", 3))
        self.tau = float(getattr(cfg, "spec_tau", 0.1))
        self.enabled = bool(getattr(cfg, "speculate", False))
        self.reset()

    def reset(self) -> None:
        """Clear any pending speculative trajectories and counters."""

        self._pending: Optional[dict] = None
        self.accepted = 0
        self.missed = 0
        self.last_distances: List[float] = []

    def has_pending(self) -> bool:
        """Return ``True`` if speculative trajectories are queued."""

        return self._pending is not None and bool(self._pending.get("branches"))

    @torch.no_grad()
    def schedule(self, z_t: torch.Tensor, action_t: torch.Tensor, task=None) -> None:
        """Start speculative rollouts from the predicted ``z_{t+1}``.

        Args:
            z_t: Latent state at time ``t``.
            action_t: Action executed in the environment at time ``t``.
            task: Optional task identifier for multi-task setups.
        """

        if not self.enabled or self.spec_depth <= 0 or self.beam_width <= 0:
            self._pending = None
            return

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        z_t = z_t.to(self.device)
        action_t = action_t.to(self.device)

        # Predict the next latent given the executed action, then roll out futures.
        pred_next = self.model.next(z_t, action_t, task)
        branches = self._speculate_from(pred_next, task)
        self._pending = {"branches": branches, "task": task}

    @torch.no_grad()
    def consume(self, z_real: torch.Tensor, task=None) -> Optional[dict]:
        """Match a real latent to speculative branches.

        Args:
            z_real: Encoded latent for the newly arrived observation ``s_{t+1}``.
            task: Optional task identifier for multi-task setups.

        Returns:
            Optional dictionary containing the selected action, the remaining
            speculative actions, and match metadata. ``None`` is returned when
            no speculative plan is available or speculation is disabled.
        """

        if not self.enabled or not self.has_pending():
            return None

        z_real = z_real.squeeze(0)
        branches: Sequence[SpeculativeBranch] = self._pending["branches"]
        task = task if task is not None else self._pending.get("task")

        distances = []
        for branch in branches:
            pred = branch.z_sequence[0].squeeze(0)
            distances.append(torch.norm(z_real - pred).item())

        self.last_distances = distances
        best_idx = int(torch.tensor(distances).argmin().item())
        best_branch = branches[best_idx]
        min_distance = distances[best_idx]
        accepted = min_distance < self.tau
        if accepted:
            self.accepted += 1
        else:
            self.missed += 1

        result = {
            "action": best_branch.actions[0],
            "remainder": best_branch.actions[1:],
            "accepted": accepted,
            "distance": min_distance,
            "task": task,
            "z_pred": best_branch.z_sequence[0],
            "branch": best_branch,
        }

        # Clear pending speculation regardless of match quality.
        self._pending = None
        return result

    @torch.no_grad()
    def _speculate_from(self, z_start: torch.Tensor, task=None) -> List[SpeculativeBranch]:
        """Roll out speculative futures starting from ``z_{t+1}``."""

        root_cache = self.model.init_dyn_cache() if self.model.transformer_dynamic else None
        beam: List[SpeculativeBranch] = [
            SpeculativeBranch(
                z_sequence=[z_start],
                actions=[],
                cumulative_reward=torch.zeros(1, device=self.device),
                dyn_cache=root_cache,
            )
        ]

        for _ in range(self.spec_depth):
            candidates: List[SpeculativeBranch] = []
            for branch in beam:
                current_z = branch.z_sequence[-1]
                if current_z.ndim == 2 and current_z.shape[0] == 1:
                    current_z = current_z
                elif current_z.ndim == 1:
                    current_z = current_z.unsqueeze(0)
                repeated_z = current_z.repeat(self.beam_width, 1)
                actions, info = self.model.pi(repeated_z, task)

                # Prefer deterministic action for the first candidate to reduce variance.
                actions = actions.clone()
                if "mean" in info:
                    actions[0] = info["mean"][0]

                for idx in range(actions.shape[0]):
                    act = actions[idx].unsqueeze(0)
                    if self.model.transformer_dynamic:
                        next_z, next_cache = self.model.next(
                            current_z, act, task, cache=branch.dyn_cache, return_cache=True
                        )
                    else:
                        next_z = self.model.next(current_z, act, task)
                        next_cache = None
                    reward_pred = self.model.reward(current_z, act, task)
                    reward_val = math.two_hot_inv(reward_pred, self.cfg)
                    cumulative = branch.cumulative_reward + reward_val
                    candidates.append(
                        SpeculativeBranch(
                            z_sequence=branch.z_sequence + [next_z],
                            actions=branch.actions + [act.squeeze(0)],
                            cumulative_reward=cumulative,
                            dyn_cache=next_cache,
                        )
                    )

            # Keep the top-k candidates by predicted cumulative reward.
            if not candidates:
                break
            rewards = torch.stack([c.cumulative_reward for c in candidates]).view(-1)
            k = min(self.beam_width, len(candidates))
            topk = torch.topk(rewards, k=k).indices
            beam = [candidates[i] for i in topk.tolist()]

        return beam
