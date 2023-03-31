import sys
from collections import abc
from typing import Dict, Optional
import logging

import pandas as pd
import numpy as np
import evaluation
import scipy

from covariance_model import get_covariance
from optimizer import get_positions
from price_model import get_prices

# TODO: fill in
NAMES_TO_FILE_NAMES = {
    "prices": "example_prices.csv",  # TODO: change
}

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def main(pricing_model_name: str, covariance_model_name: str, save_path: Optional[str] = None) -> int:
    datas: Dict[str, pd.DataFrame] = _load_data()

    datas['predicted_prices'] = get_prices(
        pricing_model_name=pricing_model_name,
        datas=datas,
    )

    datas['covariance'] = get_covariance(
        covariance_model_name=covariance_model_name,
        datas=datas,
    )

    positions = get_positions(
        datas=datas,
    )

    _save(positions)

    return 0


def _load_data() -> abc.Mapping[str, pd.DataFrame]:
    """Load data from files."""
    _log.info(f'Loading data from {NAMES_TO_FILE_NAMES.keys()}')
    return {
        name: pd.read_csv(file_name)
        for name, file_name in NAMES_TO_FILE_NAMES.items()
    }


def _save(positions: pd.DataFrame, save_path: Optional[str]) -> None:

    if save_path:
        _log.info(f'Saving positions to {save_path}')
        positions.to_csv(save_path)


if __name__ == '__main__':
    pricing_model_name = ""
    covariance_model_name = ""

    sys.exit(
        main(pricing_model_name, covariance_model_name)
    )