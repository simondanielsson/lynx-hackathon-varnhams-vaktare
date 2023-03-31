from collections import abc
import logging

import pandas as pd
import numpy as np
import scipy

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

def get_positions(datas: abc.Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Get portfolio positions using for each date.

    :param datas:
    :return: dataframe with positions for each asset for every date.
    """
    _log.info('Calculating positions...')
    pass