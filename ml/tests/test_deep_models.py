"""Phase 6 model architectures: shape correctness, parameter counts,
frozen-layer behaviour. Requires torch -- skips cleanly if unavailable."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ml.geostrom_ml.classification.deep.models import (  # noqa: E402
    N_CLASSES, SmallCNN, build_model, build_resnet18_grayscale, count_trainable_parameters,
)


class TestSmallCNN:
    def test_forward_pass_output_shape(self):
        model = SmallCNN()
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, N_CLASSES)

    def test_accepts_single_channel_input_only(self):
        model = SmallCNN()
        assert model.features[0].in_channels == 1

    def test_parameter_count_is_small(self):
        """'Start conservatively' -- verify the model really is small, not
        just documented as such."""
        model = SmallCNN()
        assert model.n_parameters() < 1_000_000

    def test_deterministic_forward_pass_in_eval_mode(self):
        model = SmallCNN().eval()
        x = torch.randn(1, 1, 224, 224)
        a = model(x)
        b = model(x)
        assert torch.allclose(a, b)


class TestResNet18Grayscale:
    def test_first_conv_accepts_one_channel(self):
        model = build_resnet18_grayscale(pretrained=False)
        assert model.conv1.in_channels == 1

    def test_output_layer_matches_n_classes(self):
        model = build_resnet18_grayscale(pretrained=False)
        assert model.fc.out_features == N_CLASSES

    def test_forward_pass_output_shape(self):
        model = build_resnet18_grayscale(pretrained=False)
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, N_CLASSES)

    def test_backbone_layers_before_freeze_point_are_frozen(self):
        model = build_resnet18_grayscale(pretrained=False, freeze_until_layer="layer4")
        assert not any(p.requires_grad for p in model.layer1.parameters())
        assert not any(p.requires_grad for p in model.layer2.parameters())
        assert not any(p.requires_grad for p in model.layer3.parameters())

    def test_layer4_and_head_remain_trainable(self):
        model = build_resnet18_grayscale(pretrained=False, freeze_until_layer="layer4")
        assert any(p.requires_grad for p in model.layer4.parameters())
        assert any(p.requires_grad for p in model.fc.parameters())

    def test_freezing_reduces_trainable_parameter_count(self):
        frozen_trainable, total = count_trainable_parameters(
            build_resnet18_grayscale(pretrained=False, freeze_until_layer="layer4"))
        full_trainable, _ = count_trainable_parameters(
            build_resnet18_grayscale(pretrained=False, freeze_until_layer=""))
        assert frozen_trainable < full_trainable == total


class TestBuildModel:
    def test_build_model_dispatches_correctly(self):
        assert isinstance(build_model("small_cnn"), SmallCNN)

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError):
            build_model("not_a_real_model")
