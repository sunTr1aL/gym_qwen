from __future__ import annotations

from typing import Dict, List, Optional

import torch


class CorrectorBuffer:
        """Lightweight replay buffer for corrector training tuples."""

        def __init__(self, capacity: int, device: Optional[torch.device] = None) -> None:
                self.capacity = int(capacity)
                self.device = device or torch.device("cpu")
                self.reset()

        def reset(self) -> None:
                self._z_real: List[torch.Tensor] = []
                self._z_pred: List[torch.Tensor] = []
                self._a_spec: List[torch.Tensor] = []
                self._a_teacher: List[torch.Tensor] = []
                self._accepted: List[bool] = []
                self._distance: List[float] = []
                self._idx = 0

        def __len__(self) -> int:
                return len(self._z_real)

        def add(
                self,
                z_real: torch.Tensor,
                z_pred: torch.Tensor,
                a_spec: torch.Tensor,
                a_teacher: torch.Tensor,
                accepted: bool,
                distance: float,
        ) -> None:
                """Store a new tuple, overwriting oldest entries when full."""

                if len(self) < self.capacity:
                        self._z_real.append(z_real.detach().cpu())
                        self._z_pred.append(z_pred.detach().cpu())
                        self._a_spec.append(a_spec.detach().cpu())
                        self._a_teacher.append(a_teacher.detach().cpu())
                        self._accepted.append(bool(accepted))
                        self._distance.append(float(distance))
                else:
                        slot = self._idx % self.capacity
                        self._z_real[slot] = z_real.detach().cpu()
                        self._z_pred[slot] = z_pred.detach().cpu()
                        self._a_spec[slot] = a_spec.detach().cpu()
                        self._a_teacher[slot] = a_teacher.detach().cpu()
                        self._accepted[slot] = bool(accepted)
                        self._distance[slot] = float(distance)
                        self._idx += 1

        def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
                """Sample a batch uniformly at random."""

                assert len(self) > 0, "Cannot sample from an empty buffer"
                batch_size = min(batch_size, len(self))
                idx = torch.randint(0, len(self), (batch_size,))
                return {
                        "z_real": torch.stack([self._z_real[i] for i in idx]).to(self.device),
                        "z_pred": torch.stack([self._z_pred[i] for i in idx]).to(self.device),
                        "a_spec": torch.stack([self._a_spec[i] for i in idx]).to(self.device),
                        "a_teacher": torch.stack([self._a_teacher[i] for i in idx]).to(self.device),
                        "accepted": torch.tensor([self._accepted[i] for i in idx], device=self.device),
                        "distance": torch.tensor([self._distance[i] for i in idx], device=self.device),
                }

        def state_dict(self) -> Dict[str, List]:
                return {
                        "z_real": self._z_real,
                        "z_pred": self._z_pred,
                        "a_spec": self._a_spec,
                        "a_plan": self._a_spec,
                        "a_teacher": self._a_teacher,
                        "accepted": self._accepted,
                        "distance": self._distance,
                }

        def load_state_dict(self, state: Dict[str, List]) -> None:
                self._z_real = list(state.get("z_real", []))
                self._z_pred = list(state.get("z_pred", []))
                self._a_spec = list(state.get("a_spec", state.get("a_plan", [])))
                self._a_teacher = list(state.get("a_teacher", []))
                self._accepted = list(state.get("accepted", []))
                self._distance = list(state.get("distance", []))
                self._idx = len(self._z_real)

        def save(self, path: str) -> None:
                torch.save(self.state_dict(), path)

        def load(self, path: str) -> None:
                state = torch.load(path, map_location=self.device)
                self.load_state_dict(state)
