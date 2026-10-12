import torch

from model import kernelGVAE


def build_model(*, corrected: bool, autoencoder: bool = False) -> kernelGVAE:
    return kernelGVAE(
        ker=torch.nn.Identity(),
        encoder=torch.nn.Identity(),
        decoder=torch.nn.Identity(),
        AutoEncoder=autoencoder,
        graphEmDim=2,
        correct_reparameterization=corrected,
    )


def test_corrected_reparameterization_scales_noise_by_standard_deviation(
    monkeypatch,
):
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    model = build_model(corrected=True)
    mean = torch.tensor([[1.0, -1.0]])
    log_std = torch.log(torch.tensor([[0.5, 2.0]]))

    sample = model.reparameterize(mean, log_std)

    torch.testing.assert_close(sample, torch.tensor([[1.5, 1.0]]))


def test_legacy_reparameterization_remains_available_for_kia_baselines(
    monkeypatch,
):
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    model = build_model(corrected=False)
    mean = torch.tensor([[1.0, -1.0]])
    log_std = torch.log(torch.tensor([[0.5, 2.0]]))

    sample = model.reparameterize(mean, log_std)

    torch.testing.assert_close(sample, torch.tensor([[1.25, 3.0]]))


def test_legacy_reparameterization_is_the_backward_compatible_default(
    monkeypatch,
):
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    model = kernelGVAE(
        ker=torch.nn.Identity(),
        encoder=torch.nn.Identity(),
        decoder=torch.nn.Identity(),
        AutoEncoder=False,
        graphEmDim=1,
    )

    sample = model.reparameterize(
        mean=torch.zeros(1, 1),
        log_std=torch.log(torch.full((1, 1), 0.5)),
    )

    torch.testing.assert_close(sample, torch.full((1, 1), 0.25))


def test_autoencoder_mode_bypasses_both_sampling_paths(monkeypatch):
    def fail_if_called(_):
        raise AssertionError("Autoencoder mode must not sample posterior noise.")

    monkeypatch.setattr(torch, "randn_like", fail_if_called)
    model = build_model(corrected=True, autoencoder=True)
    mean = torch.tensor([[1.0, -1.0]])
    log_std = torch.zeros_like(mean)

    sample = model.reparameterize(mean, log_std)

    assert sample is mean
