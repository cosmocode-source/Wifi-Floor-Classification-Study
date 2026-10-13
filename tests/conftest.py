"""Synthetic test fixtures for unit testing shared infrastructure.

IMPORTANT: All fixtures in this module generate clearly labeled SYNTHETIC data for fast,
deterministic unit testing of contracts and arithmetic. They do NOT represent measured results
from the official UJIIndoorLoc dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.config import METADATA_COLUMNS, NUM_WAP_FEATURES, UNDETECTED_SENTINEL, WAP_COLUMNS

SYNTHETIC_FIXTURE_BANNER = "SYNTHETIC_TEST_FIXTURE_ONLY_NOT_REAL_UJIINDOORLOC_DATA"


@pytest.fixture()
def synthetic_uji_df() -> pd.DataFrame:
    """Create a clearly labeled synthetic UJIIndoorLoc-like DataFrame covering Buildings 0, 1, and 2.

    - Building 0: floors 0..3 (15 rows per floor = 60 rows)
    - Building 1: floors 0..3 (15 rows per floor = 60 rows)
    - Building 2: floors 0..4 (15 rows per floor = 75 rows)
    Total = 195 synthetic rows, 529 columns (520 WAPs + 9 metadata columns).
    Includes sentinel 100, genuine 0 dBm, and negative RSSI values in [-104, -1].
    """
    rng = np.random.default_rng(12345)
    building_floors = {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3, 4],
    }
    rows_per_floor = 15

    bldg_list: list[int] = []
    floor_list: list[int] = []
    for b_id, floors in building_floors.items():
        for fl in floors:
            bldg_list.extend([b_id] * rows_per_floor)
            floor_list.extend([fl] * rows_per_floor)

    total_rows = len(bldg_list)
    wap_matrix = np.full((total_rows, NUM_WAP_FEATURES), UNDETECTED_SENTINEL, dtype=int)

    for i in range(total_rows):
        b_id = bldg_list[i]
        fl = floor_list[i]
        # Activate a deterministic subset of WAPs with realistic RSSI in [-95, 0]
        active_start = (b_id * 40 + fl * 8) % (NUM_WAP_FEATURES - 20)
        wap_matrix[i, active_start : active_start + 12] = rng.integers(-95, -20, size=12)
        # Include a genuine 0 dBm reading on one WAP for the first row of each building
        if i % 60 == 0:
            wap_matrix[i, 0] = 0

    data: dict[str, np.ndarray] = {
        col: wap_matrix[:, idx] for idx, col in enumerate(WAP_COLUMNS)
    }
    data["LONGITUDE"] = np.linspace(-7500.0, -7300.0, total_rows)
    data["LATITUDE"] = np.linspace(4864700.0, 4865000.0, total_rows)
    data["FLOOR"] = np.asarray(floor_list, dtype=int)
    data["BUILDINGID"] = np.asarray(bldg_list, dtype=int)
    data["SPACEID"] = np.arange(100, 100 + total_rows, dtype=int)
    data["RELATIVEPOSITION"] = np.ones(total_rows, dtype=int)
    data["USERID"] = np.ones(total_rows, dtype=int)
    data["PHONEID"] = np.full(total_rows, 13, dtype=int)
    data["TIMESTAMP"] = np.arange(1371710000, 1371710000 + total_rows, dtype=int)

    df = pd.DataFrame(data)
    assert list(df.columns) == [*WAP_COLUMNS, *METADATA_COLUMNS]
    df.attrs["synthetic_fixture"] = SYNTHETIC_FIXTURE_BANNER
    return df
