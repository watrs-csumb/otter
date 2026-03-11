from pathlib import Path
from otter import water_balance as wb

import pandas as pd
import pytest

import numpy.testing as npt


@pytest.mark.parametrize("dataset", list(Path("./tests/static/wil/").glob("pt_*.csv")))
def test_interp_et(dataset):
    df = pd.read_csv(dataset, parse_dates=["dt"])
    nodata = -9999.0

    et_arr = df["et"].to_numpy(dtype="float32")
    eto_arr = df["eto"].to_numpy(dtype="float32")
    et_interp_expected = df["et_interp"].to_numpy(dtype="float32")

    et_interp = wb.etof_interp(et_arr, eto_arr, nodata=nodata)
    
    npt.assert_allclose(et_interp, et_interp_expected, strict=True, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("dataset", list(Path("./tests/static/wil/").glob("pt_*.csv")))
def test_wb_interp(dataset):
    df = pd.read_csv(dataset, parse_dates=["dt"])
    nodata = -9999.0
    aws_max = df["aws_max"].to_numpy(dtype="float32")
    aws_u_ts = df["aws_u"].to_numpy(dtype="float32")
    cn_ts = df["cn"].to_numpy(dtype="float32")
    pr_ts = df["pr"].to_numpy(dtype="float32")
    et_ts = df["et"].to_numpy(dtype="float32")
    eto_ts = df["eto"].to_numpy(dtype="float32")
    res = wb.do_wb_interp(
        aws_max.astype("float32")[0],
        aws_u_ts,
        cn_ts,
        pr_ts,
        et_ts,
        eto_ts,
        nodata=nodata,
    )
    (dru, drl, perc, dperc, ro, etaw, peff, et) = res

    expected = {
        "dru": df["dru"].to_numpy(dtype="float32"),
        "drl": df["drl"].to_numpy(dtype="float32"),
        "perc": df["perc"].to_numpy(dtype="float32"),
        "dperc": df["dperc"].to_numpy(dtype="float32"),
        "ro": df["ro"].to_numpy(dtype="float32"),
        "etaw": df["etaw"].to_numpy(dtype="float32"),
        "peff": df["peff"].to_numpy(dtype="float32"),
        "et": df["et_interp"].to_numpy(dtype="float32"),
    }

    npt.assert_allclose(dru, expected["dru"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(drl, expected["drl"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(perc, expected["perc"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(dperc, expected["dperc"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(ro, expected["ro"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(etaw, expected["etaw"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(peff, expected["peff"], strict=True, rtol=1e-1, atol=1e-1)
    npt.assert_allclose(et, expected["et"], strict=True, rtol=1e-1, atol=1e-1)
