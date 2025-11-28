import pytest

torch = pytest.importorskip("torch")

from speculative_manager import SpeculativeBranch, SpeculativeManager


class DummyCfg:
        def __init__(self, **kwargs):
                # Provide defaults required by SpeculativeManager / math.two_hot_inv
                self.num_bins = kwargs.get("num_bins", 1)
                self.vmin = kwargs.get("vmin", -1.0)
                self.vmax = kwargs.get("vmax", 1.0)
                self.bin_size = kwargs.get("bin_size", 0.5)
                self.speculate = kwargs.get("speculate", True)
                self.beam_width = kwargs.get("beam_width", 3)
                self.spec_depth = kwargs.get("spec_depth", 2)
                self.spec_tau = kwargs.get("spec_tau", 0.25)

        def get(self, key, default=None):
                return getattr(self, key, default)


class DummyModel:
        transformer_dynamic = False

        def __init__(self, action_dim=1):
                self.action_dim = action_dim

        def init_dyn_cache(self):
                return None

        def encode(self, obs, task=None):
                return obs

        def next(self, z, a, task=None, cache=None, return_cache=False):
                next_z = z + a
                return (next_z, cache) if return_cache else next_z

        def reward(self, z, a, task=None):
                # Deterministic reward that scales with the latent to simplify assertions
                return torch.ones(z.shape[0], 1) * torch.norm(z + a, dim=-1, keepdim=True)

        def pi(self, z, task=None):
                # Centered policy with tiny stochasticity
                mean = torch.zeros_like(z)
                return mean, {"mean": mean, "entropy": torch.zeros_like(mean)}


def test_beam_generation_depth_and_width():
        cfg = DummyCfg()
        model = DummyModel()
        manager = SpeculativeManager(model, cfg, torch.device("cpu"))
        start_z = torch.zeros(1, 1)
        action = torch.zeros(1, 1)

        manager.schedule(start_z, action)
        assert manager.has_pending()
        branches = manager._pending["branches"]
        assert len(branches) == cfg.beam_width
        for branch in branches:
                assert len(branch.z_sequence) == cfg.spec_depth + 1
                assert len(branch.actions) == cfg.spec_depth


def test_branch_selection_matches_closest_prediction():
        cfg = DummyCfg(spec_tau=0.2)
        model = DummyModel()
        manager = SpeculativeManager(model, cfg, torch.device("cpu"))
        # Manually craft branches
        branch_a = SpeculativeBranch(
                z_sequence=[torch.tensor([[0.1]])], actions=[torch.tensor([0.0])], cumulative_reward=torch.tensor([0.0])
        )
        branch_b = SpeculativeBranch(
                z_sequence=[torch.tensor([[1.0]])], actions=[torch.tensor([1.0])], cumulative_reward=torch.tensor([0.0])
        )
        manager._pending = {"branches": [branch_a, branch_b], "task": None}

        z_real = torch.tensor([[0.05]])
        result = manager.consume(z_real)

        assert result is not None and result["accepted"]
        assert torch.allclose(result["action"], branch_a.actions[0])
        assert torch.allclose(result["z_pred"], branch_a.z_sequence[0])


def test_speculative_reuse_reduces_planning_calls():
        class Harness:
                def __init__(self):
                        self.cfg = DummyCfg(spec_tau=0.5)
                        self.model = DummyModel()
                        self.spec_manager = SpeculativeManager(self.model, self.cfg, torch.device("cpu"))
                        self.plan_calls = 0

                def plan(self, obs):
                        self.plan_calls += 1
                        action = torch.zeros(1, 1)
                        self.spec_manager.schedule(obs, action)
                        return action

                def act(self, obs, t0=False):
                        if t0:
                                self.spec_manager.reset()
                        z = self.model.encode(obs)
                        if self.spec_manager.has_pending():
                                match = self.spec_manager.consume(z)
                                if match and match["accepted"]:
                                        return match["action"], False
                        action = self.plan(z)
                        return action.squeeze(0), True

        harness = Harness()
        obs0 = torch.zeros(1, 1)
        action0, planned0 = harness.act(obs0, t0=True)
        assert planned0 and harness.plan_calls == 1

        # Next observation matches the prediction, so the speculative branch is accepted
        obs1 = obs0 + action0
        action1, planned1 = harness.act(obs1)
        assert not planned1
        assert harness.plan_calls == 1
        assert torch.allclose(action1, torch.zeros(1))
