# make_cn
from collections.abc import Generator
from importlib import resources
from os import remove
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from osgeo import gdal
from pytest import fixture, raises

from otter import soils, static

@fixture(scope="module")
def static_files():
    static_set = {}

    static_path = resources.files(static)
    with resources.as_file(static_path / "cdl_rz_cn.csv") as f:
        static_set["cdl_rz_cn"] = pd.read_csv(f)
    with resources.as_file(static_path / "gnatsgo.csv") as f:
        static_set["gnatsgo"] = pd.read_csv(f)
    with resources.as_file(static_path / "cn_table.csv") as f:
        static_set["cn_table"] = pd.read_csv(f)

    yield static_set
    

