# make_cn
from collections.abc import Generator
from importlib import resources
from os import remove
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from pytest import fixture, raises

from otter import soils, static
import pytest
    
# valid mukey: 49315
# invalid mukey: 0

def test_calc_cn_success():
    # Code logic results in a successful passthrough returning a non-zero integer.
    crop_dict = soils.get_crop_cat_dict()
    hydgrp_dict = soils.get_hydgrp_dict()
    cn_dict = soils.get_cn_dict()

    cn = soils.calc_cn(cdl_code=1, mukey=49315, crop_cat_dict=crop_dict, hydgrp_dict=hydgrp_dict, cn_dict=cn_dict)

    assert cn > 0

def test_calc_cn_fail_nullmukey():
    crop_dict = soils.get_crop_cat_dict()
    hydgrp_dict = soils.get_hydgrp_dict()
    cn_dict = soils.get_cn_dict()

    cn = soils.calc_cn(cdl_code=1, mukey=0, crop_cat_dict=crop_dict, hydgrp_dict=hydgrp_dict, cn_dict=cn_dict)

    assert cn == 0

def test_calc_aws_success():
    rz_dict = soils.get_rz_dict()
    aws_dict = soils.get_aws_dict()

    aws = soils.calc_aws(cdl_code=1, mukey=49315, rz_dict=rz_dict, aws_dict=aws_dict)

    assert aws > 0

def test_calc_aws_fail_nullmukey():
    rz_dict = soils.get_rz_dict()
    aws_dict = soils.get_aws_dict()

    aws = soils.calc_aws(cdl_code=1, mukey=0, rz_dict=rz_dict, aws_dict=aws_dict)

    assert aws == 0

def test_make_hydgrp_dict_success():
    fake_hygrp_array = np.array(["D", "C", "A", "D", "B"])
    fake_mukey_array = np.array([49316, 49415, 50432, 51298, 52912])

    hydgrp_dict = soils.make_hydgrp_dict(fake_mukey_array, fake_hygrp_array)

    assert hydgrp_dict == {
        49316: 4,
        49415: 3,
        50432: 1, 
        51298: 4, 
        52912: 2
    }

@pytest.mark.parametrize("cdl", [1, 2, 4])
@pytest.mark.parametrize("mukey", [49315, 51096, 51693])
@pytest.mark.parametrize("make_max", [False, True])
def test_pt_soil_func(cdl, mukey, make_max):
    pt_soil = soils.pt_soil_func()

    # make_max=False returns (aws, cn)
    if not make_max:
        aws, cn = pt_soil(cdl_code=cdl, mukey=mukey) # type: ignore

        assert aws > 0 and cn > 0 and cn <= 100
    
    # make_max=True returns (aws, aws_max, cn)
    else:
        aws, aws_max, cn = pt_soil(cdl_code=cdl, mukey=mukey, make_max=make_max) # type: ignore

        assert cn is not None   # <-- handled make_max correctly
        assert aws > 0 and aws < aws_max and cn > 0 and cn <= 100

