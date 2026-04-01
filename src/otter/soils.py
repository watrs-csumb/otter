from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from numba.typed import Dict

from . import static

static_path = resources.files(static)


######### AWS #########


@njit
def make_aws_dict(mukey, aws25, aws50, aws100, aws150):
    d = dict()
    for i in range(mukey.size):
        d[mukey[i]] = np.array([aws25[i], aws50[i], aws100[i], aws150[i]])

    return d


@njit
def make_rz_dict(crop_code, depth):
    d = dict()
    for i in range(crop_code.size):
        d[crop_code[i]] = depth[i]

    # set max depth to code -1
    # background (code 0) has nan depth
    d[-1] = np.nanmax(depth)

    return d


@njit
def calc_aws(cdl_code, mukey, rz_dict, aws_dict):
    if cdl_code == 0 or mukey == 0:
        return 0

    try:
        rz_depth = rz_dict[cdl_code]
    except Exception:
        return 0
    if np.isnan(rz_depth):
        return 0

    aws25, aws50, aws100, aws150 = aws_dict[mukey]

    if rz_depth < 25:
        aws = aws25 * rz_depth / 25
    elif rz_depth >= 25 and rz_depth < 50:
        marginal_frac = (rz_depth - 25) / 25
        marginal_aws = marginal_frac * (aws50 - aws25)
        aws = aws25 + marginal_aws
    elif rz_depth >= 50 and rz_depth < 100:
        marginal_frac = (rz_depth - 50) / 50
        marginal_aws = marginal_frac * (aws100 - aws50)
        aws = aws50 + marginal_aws
    elif rz_depth >= 100 and rz_depth < 150:
        marginal_frac = (rz_depth - 100) / 50
        marginal_aws = marginal_frac * (aws150 - aws100)
        aws = aws100 + marginal_aws
    else:
        rz_beyond_150 = rz_depth - 150
        deepest_awc = (aws150 - aws100) / 50
        aws = aws150 + rz_beyond_150 * deepest_awc

    # return value in mm
    return aws * 10


######### CN #########


@njit
def make_hydgrp_dict(mukey: np.ndarray, hydgrp: np.ndarray):
    d = dict()
    for i in range(mukey.size):
        if hydgrp[i] == "A":
            d[mukey[i]] = 1
        elif hydgrp[i] == "B":
            d[mukey[i]] = 2
        elif hydgrp[i] == "C":
            d[mukey[i]] = 3
        # elif np.string_.find(hydgrp[i], "D") != -1:
        elif "D" in hydgrp[i]:
            d[mukey[i]] = 4
        elif hydgrp[i] == "":
            d[mukey[i]] = 0
        else:
            print(f"Row {i} not in ABC or empty in make_hygrp_dict")

    return d


@njit
def make_crop_cat_dict(crop_code, cn_category):
    d = dict()
    for i in range(crop_code.size):
        d[crop_code[i]] = cn_category[i]

    return d


@njit
def make_cn_dict(cn_category: np.ndarray, vals: np.ndarray) -> dict[int, np.ndarray]:
    d = dict()
    for i in range(cn_category.size):
        d[cn_category[i]] = vals[i, :]

    return d


@njit
def calc_cn(
    cdl_code: int,
    mukey: int,
    crop_cat_dict: dict[int, int],
    hydgrp_dict: dict[int, int],
    cn_dict: dict[int, np.ndarray]
) -> int:
    if cdl_code == 0 or mukey == 0:
        return 0

    crop_cat = crop_cat_dict[cdl_code]
    if crop_cat == -1.0:
        return 0

    hydgrp = hydgrp_dict[mukey]
    if hydgrp == 0:
        return 0

    # hydgrp is 1-4, so subtract 1 for python indexing
    # but just in case, throw an error if hydgrp is less than 1
    if hydgrp - 1 < 0:
        raise ValueError(f"Invalid hydgrp {hydgrp}")
    
    return cn_dict[crop_cat][hydgrp - 1]



######### utils #########


def get_aws_dict() -> Dict:
    aws_table_path = static_path / "gnatsgo.csv"
    with resources.as_file(aws_table_path) as aws_table_fn:
        aws_df = pd.read_csv(aws_table_fn)

    aws_dict = make_aws_dict(
        np.array(aws_df.mukey),
        np.array(aws_df.aws025wta),
        np.array(aws_df.aws050wta),
        np.array(aws_df.aws0100wta),
        np.array(aws_df.aws0150wta),
    )
    return aws_dict


def get_rz_dict() -> Dict:
    crop_rz_path = static_path / "cdl_rz_cn.csv"
    with resources.as_file(crop_rz_path) as crop_rz_fn:
        crop_df = pd.read_csv(crop_rz_fn)

    crop_df["cdl_code"] = crop_df["CDL code"].astype("int")
    crop_df["depth"] = crop_df["Rooting depth (m)"] * 100
    mrd = np.array(crop_df.depth)
    rz_dict = make_rz_dict(np.array(crop_df.cdl_code), mrd)
    return rz_dict


def get_cn_dict() -> Dict:
    cn_table_path = static_path / "cn_table.csv"
    with resources.as_file(cn_table_path) as cn_table_fn:
        cn_df = pd.read_csv(cn_table_fn)

    grp_cols = np.array(cn_df[["a", "b", "c", "d"]])
    cn_dict = make_cn_dict(np.array(cn_df.cn_category), grp_cols)
    return cn_dict


def get_hydgrp_dict() -> Dict:
    hydgrp_path = static_path / "gnatsgo.csv"
    with resources.as_file(hydgrp_path) as hydgrp_fn:
        hydgrp_df = pd.read_csv(hydgrp_fn)
        hydgrp_df = hydgrp_df[["mukey", "hydgrpdcd"]]

    hydgrp_arr = np.array(hydgrp_df.hydgrpdcd.fillna(""), dtype=str)
    hydgrp_dict = make_hydgrp_dict(np.array(hydgrp_df.mukey), hydgrp_arr)
    return hydgrp_dict


def get_crop_cat_dict() -> Dict:
    crop_rz_path = static_path / "cdl_rz_cn.csv"
    with resources.as_file(crop_rz_path) as crop_rz_fn:
        crop_df = pd.read_csv(crop_rz_fn)
    crop_df["cdl_code"] = crop_df["CDL code"].astype("int")

    # no reason to jit this but imported numba so why not
    @njit
    def cn_cat(cat):
        return -1 if np.isnan(cat) else int(cat)

    crop_df["cn_category"] = crop_df["CN category"].apply(cn_cat)

    crop_cat_dict = make_crop_cat_dict(
        np.array(crop_df.cdl_code, dtype=int), np.array(crop_df.cn_category, dtype=int)
    )
    return crop_cat_dict


def pt_soil_func():
    aws_dict = get_aws_dict()
    rz_dict = get_rz_dict()
    cn_dict = get_cn_dict()
    hydgrp_dict = get_hydgrp_dict()
    crop_cat_dict = get_crop_cat_dict()

    def pt_soil(cdl_code, mukey, make_max=False):
        aws = calc_aws(cdl_code, mukey, rz_dict, aws_dict)
        cn = calc_cn(cdl_code, mukey, crop_cat_dict, hydgrp_dict, cn_dict)

        if not make_max:
            return aws, cn

        # -1 means use max possible depth
        aws_max = calc_aws(-1, mukey, rz_dict, aws_dict)
        return aws, aws_max, cn

    return pt_soil
