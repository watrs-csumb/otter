### OTTER Testing Utilities ###
#
# Pytest uses conftest.py to define package-wide fixtures (simple, consistent contexts)
#       i.e. database environment setup, dataset content
#
# Modules to be tested SHOULD NOT have a conftest fixture made at this level.
# Any fixture made with a tested module should remain within that module's test suite.
#
#

import logging
import sys

from collections.abc import Generator
from os import chdir, getcwd
from pandas import read_csv
from pathlib import Path 
from pytest import fixture
from tempfile import TemporaryDirectory

@fixture(scope="session")
def logger():
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[stdout_handler],
    )

    log = logging.getLogger("Pytest")
    
    yield log

@fixture(scope="module")
def datasets(cleandir):
    """
    Provides a dictionary containing data objects on a per-module basis.
    """
    cwd = Path(cleandir)
    datasets = {}

    try:
        area_ts_dir = cwd / "tests/static/wil"
        area_ts_fns = area_ts_dir.glob("pt_*.csv")
        datasets["area_ts_dfs"] =\
            [read_csv(fn, parse_dates=["dt"]) for fn in area_ts_fns]

    except:
        print("Error in constructing dataset paths! Are you in the correct directory?")
        raise
    
    yield datasets

    datasets["mukey_ras"].Close()
    datasets["area_cdl_ras"].Close()

@fixture(scope="module")
def cleandir() -> Generator[str, None, None]:
    """
    Creates a clean, empty directory for temp outputs.\n
    Yields the original working directory for reading datasets.
    Provides clean directories on a per-module basis.
    """
    with TemporaryDirectory() as tempdir:
        original_cwd = getcwd()
        chdir(tempdir)

        yield original_cwd

        # Revert cwd to original path.
        chdir(original_cwd)

