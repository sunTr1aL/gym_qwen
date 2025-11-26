import pytest

torch = pytest.importorskip("torch")

from tdmpc2.corrector import Corrector
from tdmpc2.corrector_buffer import CorrectorBuffer


def test_corrector_applies_residual_addition():
        latent_dim, act_dim = 4, 2
        corrector = Corrector(obs_dim=latent_dim, act_dim=act_dim, latent_dim=latent_dim, tanh_output=False)
        # Replace model with a deterministic linear layer that outputs ones.
        input_dim = 3 * latent_dim + act_dim
        layer = torch.nn.Linear(input_dim, act_dim, bias=False)
        torch.nn.init.constant_(layer.weight, 0.0)
        with torch.no_grad():
                layer.weight[:, 0] = 1.0
        corrector.model = layer

        z_real = torch.zeros(1, latent_dim)
        z_pred = torch.zeros(1, latent_dim)
        a_spec = torch.zeros(1, act_dim)
        corrected = corrector(z_real, z_pred, a_spec)
        assert torch.allclose(corrected, torch.ones_like(a_spec))


def test_corrector_loss_regularizes_delta_from_spec():
        corrector = Corrector(obs_dim=2, act_dim=1, latent_dim=2, tanh_output=False)
        a_corr = torch.tensor([[0.5]])
        a_teacher = torch.tensor([[0.0]])
        a_spec = torch.tensor([[0.2]])
        loss = corrector.loss_fn(a_corr, a_teacher, reg_lambda=1.0, a_spec=a_spec)
        # Residual from speculative action is 0.3, so regularizer contributes 0.09
        expected = torch.nn.functional.mse_loss(a_corr, a_teacher) + 0.09
        assert torch.isclose(loss, expected)


def test_corrector_buffer_overwrite_and_sample():
        buffer = CorrectorBuffer(capacity=2, device=torch.device("cpu"))
        for i in range(3):
                val = torch.full((1, 2), float(i))
                buffer.add(val, val, val, val, accepted=bool(i % 2), distance=float(i))
        assert len(buffer) == 2
        batch = buffer.sample(batch_size=2)
        assert set(batch.keys()) == {"z_real", "z_pred", "a_spec", "a_teacher", "accepted", "distance"}
        assert batch["z_real"].shape[0] == 2
