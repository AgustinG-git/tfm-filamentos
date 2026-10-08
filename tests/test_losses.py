"""Se ejecutan donde haya torch y smp (Kaggle); si no, se saltan."""

import pytest

smp = pytest.importorskip("segmentation_models_pytorch")
torch = pytest.importorskip("torch")

from filseg.losses import LOSSES, cldice_loss, soft_skel  # noqa: E402


def filamento():
    y = torch.zeros(2, 1, 64, 64)
    y[:, :, 30:34, 5:60] = 1.0      # tira horizontal
    y[:, :, 10:30, 40:43] = 1.0     # barba vertical
    return y


def test_esqueleto_es_fino_y_dentro_del_filamento():
    y = filamento()
    sk = soft_skel(y, iters=10)
    assert sk.sum() > 0
    assert sk.sum() < 0.5 * y.sum()             # mucho más fino
    assert (sk * (1 - y)).sum() == 0            # no se sale


def test_cldice_perfecto_y_con_barba_perdida():
    y = filamento()
    assert cldice_loss(y, y).item() == pytest.approx(0.0, abs=1e-6)
    sin_barba = y.clone()
    sin_barba[:, :, 10:30, 40:43] = 0
    assert cldice_loss(sin_barba, y).item() > 0.05   # perder la barba cuesta


@pytest.mark.parametrize("name", list(LOSSES))
def test_todas_las_perdidas(name):
    args = {"bce_dice": {}, "bce_iou": {}, "bce_dice_cldice": {"iters": 5}, "bce_cldice": {"iters": 5},
            "bce_iou_cldice": {"iters": 5}}[name]
    loss_fn = LOSSES[name](**args)
    y = filamento()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    loss = loss_fn(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    bueno = loss_fn((y * 20 - 10).detach(), y)      # predicción casi perfecta
    assert bueno.item() < loss.item()
