import numpy as np
import pandas as pd
import pytest

from filseg.contract import FULL_SIZE
from filseg.gt import mask_to_rle


def rect(r0, r1, c0, c1):
    """Máscara 2048×2048 con un rectángulo [r0:r1, c0:c1]."""
    m = np.zeros(FULL_SIZE, np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def frame(items):
    """[(filament_id, máscara)] -> DataFrame en formato oficial."""
    return pd.DataFrame(
        [{"filament_id": fid, "segmentation_rle": mask_to_rle(m)} for fid, m in items],
        columns=["filament_id", "segmentation_rle"],
    )


@pytest.fixture
def gt_100():
    """Un filamento GT de 10×10 = 100 px."""
    return frame([("ana-img1_1", rect(0, 10, 0, 10))])
