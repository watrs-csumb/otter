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
import pytest
    
# tests to-do
# 1) mukey not having data in gnatsgo file (no corresponding hydrogp) (calc_cn, calc_aws)
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

# 2) test for mukey not being in hydrgp table
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

@pytest.mark.skip("Unfinished")
def test_make_hydrgrp_dict_fail(capsys):
    # A message is printed when mukey index is not in hydgrp array, so basically only when the lengths of the two arrays are mismatched.
    fake_hygrp_array = np.array(["D", "C", "A", "D", "B"])
    fake_mukey_array = np.array([49316, 49415, 50432, 51298, 52912, 53374])

    hydgrp_dict = soils.make_hydgrp_dict(fake_mukey_array, fake_hygrp_array)

    captured = capsys.readouterr()
    assert captured.out == "Row 5 not in ABC or empty in make_hygrp_dict"
    
    assert hydgrp_dict == {
        49316: 4,
        49415: 3,
        50432: 1, 
        51298: 4, 
        52912: 2
    }


@pytest.mark.skip("Not implemented")
# 3) pt_soil_func (use for api)
def test_pt_soil_func():
    pass
