import math

import torch

from src.models.encoders.tcn import TCNEncoder
from src.models.ssl.quantile import EncoderQuantileWrapper, QuantileHead, pinball_loss


def test_pinball_loss_at_median_equals_half_mae():
    preds = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([1.0, -1.0])
    loss = pinball_loss(preds, target, quantiles=[0.5])
    assert math.isclose(loss.item(), 0.5, rel_tol=1e-6)


def test_pinball_loss_asymmetry_at_high_quantile():
    # tau=0.9: under-prediction (target above pred) is penalized ~9x harder
    # than over-prediction of the same magnitude -- that asymmetry is the
    # entire point of quantile regression.
    under_pred = pinball_loss(torch.tensor([[0.0]]), torch.tensor([1.0]), quantiles=[0.9])
    over_pred = pinball_loss(torch.tensor([[0.0]]), torch.tensor([-1.0]), quantiles=[0.9])
    assert math.isclose(under_pred.item(), 0.9, rel_tol=1e-6)
    assert math.isclose(over_pred.item(), 0.1, rel_tol=1e-6)
    assert under_pred.item() > over_pred.item()


def test_pinball_loss_multi_quantile_shape_and_positivity():
    quantiles = [0.1, 0.5, 0.9]
    preds = torch.randn(8, len(quantiles))
    target = torch.randn(8)
    loss = pinball_loss(preds, target, quantiles)
    assert loss.dim() == 0  # scalar (mean-reduced)
    assert loss.item() >= 0


def test_quantile_head_output_shape():
    head = QuantileHead(in_features=32)
    x = torch.randn(5, 32)
    out = head(x)
    assert out.shape == (5, 9)  # README spec: 9 quantiles
    assert head.median_index() == 4


def test_quantile_head_trains_toward_correct_median():
    """Not a full convergence test (would be slow/flaky) -- just checks a
    few gradient steps move the median prediction toward the target mean,
    confirming pinball_loss's gradient sign is correct end to end."""
    torch.manual_seed(0)
    head = QuantileHead(in_features=4, quantiles=[0.5])
    x = torch.randn(64, 4)
    target = torch.full((64,), 5.0)  # constant target -> median should converge toward 5.0

    opt = torch.optim.Adam(head.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        pred = head(x)
        loss = pinball_loss(pred, target, quantiles=[0.5])
        loss.backward()
        opt.step()

    final_pred = head(x).mean().item()
    assert abs(final_pred - 5.0) < 0.5, f"median prediction {final_pred} did not converge toward target 5.0"


def test_encoder_quantile_wrapper_with_tcn():
    encoder = TCNEncoder(n_features=6, channels=16)
    model = EncoderQuantileWrapper(encoder, hidden_size=16)
    x = torch.randn(4, 30, 6)
    out = model(x)
    assert out.shape == (4, 9)
    loss = pinball_loss(out, torch.randn(4), model.head.quantiles)
    loss.backward()
    assert encoder.blocks[0].conv.weight.grad is not None, "gradient must flow back through the encoder"
