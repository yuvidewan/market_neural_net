"""
Structural tests for the two-axis Transformer encoder (README Phase 3 v2).

The precise invariant being tested (stated in the module docstring):
no patch position p's output representation, for ANY symbol, may depend on
ANY symbol's input data at a patch position > p. Perturbing input at the
LAST patch must leave every earlier patch's output bit-for-bit unchanged,
for every symbol -- not just the perturbed one -- across the FULL multi-block
stack, since a residual/reshape bug in one block could leak information the
other blocks don't (a single-block test wouldn't catch that).
"""
import torch

from src.models.encoders.transformer import PatchEmbed, TwoAxisTransformerEncoder
from src.models.ssl.quantile import EncoderQuantileWrapper


def _make_encoder(n_features=3, d_model=8, n_heads=2, n_blocks=3, patch_size=2):
    return TwoAxisTransformerEncoder(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_blocks=n_blocks, patch_size=patch_size, dropout=0.0,
    )


def test_temporal_causality_across_full_stack():
    torch.manual_seed(0)
    n_symbols, seq_len, n_features = 4, 8, 3  # patch_size=2 -> 4 patches
    encoder = _make_encoder(n_features=n_features).eval()

    x = torch.randn(n_symbols, seq_len, n_features)
    out1 = encoder(x)

    x2 = x.clone()
    x2[0, -2:, :] += 5.0  # perturb only the LAST patch's rows, symbol 0 only
    out2 = encoder(x2)

    n_patches = out1.shape[1]
    # every symbol's every patch BEFORE the perturbed one is untouched
    assert torch.allclose(out1[:, : n_patches - 1, :], out2[:, : n_patches - 1, :], atol=1e-6), (
        "a patch before the perturbation changed -- lookahead leak"
    )
    # sanity: the perturbed symbol's last patch DID change (test isn't vacuous)
    assert not torch.allclose(out1[0, -1, :], out2[0, -1, :], atol=1e-6)


def test_cross_sectional_attention_actually_mixes_symbols():
    torch.manual_seed(0)
    n_symbols, seq_len, n_features = 4, 4, 3  # patch_size=2 -> 2 patches
    encoder = _make_encoder(n_features=n_features, n_blocks=1, patch_size=2).eval()

    x = torch.randn(n_symbols, seq_len, n_features)
    out1 = encoder(x)

    x2 = x.clone()
    x2[0, -2:, :] += 5.0  # perturb symbol 0's last patch only
    out2 = encoder(x2)

    # a DIFFERENT symbol's representation at the SAME (last) patch position
    # should change too -- proof cross-sectional attention connects them,
    # not just an accidental no-op axis.
    assert not torch.allclose(out1[1, -1, :], out2[1, -1, :], atol=1e-6), (
        "perturbing symbol 0 didn't affect symbol 1 at the same patch position -- "
        "cross-sectional attention isn't mixing symbols"
    )


def test_patch_padding_puts_real_data_in_last_patch():
    # seq_len=7 not divisible by patch_size=3 -> left-padded to 9 -> 3 patches:
    # patch0=[pad,pad,row0], patch1=[row1,row2,row3], patch2=[row4,row5,row6].
    # With an all-ones projection (bias 0), a patch's output is just the sum
    # of its (patch_size * n_features) input values -- so the LAST patch,
    # built entirely from real nonzero data, must sum to more than the FIRST
    # patch, 4 of whose 6 slots are zero-padding.
    embed = PatchEmbed(n_features=2, patch_size=3, d_model=4)
    x = torch.arange(1, 7 * 2 + 1, dtype=torch.float32).reshape(1, 7, 2)  # values 1..14, all positive
    with torch.no_grad():
        embed.proj.weight.fill_(1.0)
        embed.proj.bias.fill_(0.0)
        out = embed(x)
    assert out.shape == (1, 3, 4)  # 9 // 3 = 3 patches
    assert out[0, 0, 0].item() < out[0, -1, 0].item(), (
        "first patch should be mostly zero-padding, last patch fully real data -- "
        "padding landed on the wrong (future) side"
    )
    # exact expected sums, spelled out: patch0 = 0+0+0+0+1+2 = 3; patch2 = 9+10+11+12+13+14 = 69
    assert out[0, 0, 0].item() == 3.0
    assert out[0, -1, 0].item() == 69.0


def test_output_shape():
    n_symbols, seq_len, n_features = 5, 16, 3
    encoder = _make_encoder(n_features=n_features, d_model=8, patch_size=4)
    x = torch.randn(n_symbols, seq_len, n_features)
    out = encoder(x)
    assert out.shape == (n_symbols, 4, 8)  # 16/4=4 patches, d_model=8


def test_encoder_quantile_wrapper_integration():
    """The whole point of the [batch, seq, hidden] convention: this wrapper
    is completely unmodified from the LSTM/TCN case (src/models/ssl/quantile.py)."""
    n_symbols, seq_len, n_features, d_model = 6, 8, 3, 8
    encoder = _make_encoder(n_features=n_features, d_model=d_model, patch_size=2)
    model = EncoderQuantileWrapper(encoder, hidden_size=d_model)
    x = torch.randn(n_symbols, seq_len, n_features)
    out = model(x)
    assert out.shape == (n_symbols, 9)  # 9 quantiles by default
