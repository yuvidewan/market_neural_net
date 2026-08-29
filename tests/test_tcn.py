"""
Structural causality test for the TCN encoder -- the conv-net analogue of
"an LSTM can't see the future because it's recurrent". A dilated causal
conv net doesn't get that for free from the architecture family, only from
the left-only padding in CausalTCNBlock, so this is tested directly rather
than assumed.
"""
import torch

from src.models.encoders.tcn import TCN_DILATIONS, TCN_KERNEL_SIZE, TCNBaseline, TCNEncoder


def test_receptive_field_matches_readme_spec():
    enc = TCNEncoder(n_features=4)
    assert enc.receptive_field == 256, "README spec: 8 dilations summing to 255, kernel_size=2 -> RF=256"


def test_output_shape():
    enc = TCNEncoder(n_features=6, channels=32)
    x = torch.randn(3, 50, 6)
    out = enc(x)
    assert out.shape == (3, 50, 32)


def test_causality_future_perturbation_does_not_change_past_output():
    torch.manual_seed(0)
    enc = TCNEncoder(n_features=4, channels=8)
    enc.eval()

    seq_len = 40
    x = torch.randn(2, seq_len, 4)
    with torch.no_grad():
        out_before = enc(x)

    x_perturbed = x.clone()
    x_perturbed[:, seq_len // 2 + 1:, :] += 100.0  # blow up everything strictly after the midpoint

    with torch.no_grad():
        out_after = enc(x_perturbed)

    # every position UP TO AND INCLUDING the midpoint must be bit-for-bit
    # unchanged -- it never saw the perturbed future
    assert torch.allclose(out_before[:, :seq_len // 2 + 1, :], out_after[:, :seq_len // 2 + 1, :], atol=1e-5), (
        "TCN output at time t changed after perturbing inputs at t+1 onward -- causality is broken"
    )
    # sanity: the perturbation must actually have propagated somewhere,
    # otherwise this test would trivially pass for the wrong reason
    assert not torch.allclose(out_before[:, seq_len // 2 + 1:, :], out_after[:, seq_len // 2 + 1:, :], atol=1e-5)


def test_baseline_forward_and_backward():
    model = TCNBaseline(n_features=5)
    x = torch.randn(4, 30, 5)
    y = model(x)
    assert y.shape == (4,)
    loss = y.pow(2).mean()
    loss.backward()  # gradients must flow through the whole stack without error
    assert model.encoder.blocks[0].conv.weight.grad is not None
