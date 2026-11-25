"""Reading Bruker data."""
from pathlib import Path

import nmrglue as ng


def load_bruker(path):
    """Load a Bruker experiment directory (the folder containing fid/ser).

    Returns (dic, data) with the digital filter removed for 1D data.
    """
    path = Path(path)
    dic, data = ng.bruker.read(str(path))
    data = ng.bruker.remove_digital_filter(dic, data)
    return dic, data
