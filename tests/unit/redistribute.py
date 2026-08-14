"""Tests for the crop-switch depletion redistribution in ``do_wb_interp``.

Each case is driven through ``do_wb_interp`` with zero weather forcing so the
only thing that changes state is the crop switch on day 1. Day 0 holds the
given pre-switch depletion (dru, drl); day 1 changes the upper capacity and
triggers redistribution, so dru[1]/drl[1] are the post-switch depletion.

Notation: w = aws - dr is water content; the total column capacity aws_t is
fixed, so the lower capacity is aws_t - aws_u and total column water
w_t = (aws_u - dru) + (aws_t - aws_u - drl) is conserved by both rules.
"""
import numpy as np
import pytest
import numpy.testing as npt

from otter import water_balance as wb


def _switch(dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t, rule):
    """Return post-switch (dru, drl) after one crop switch, zero forcing."""
    aws_u_ts = np.array([aws_u_prev, aws_u_new], dtype="float32")
    zeros = np.zeros(2, dtype="float32")
    dru, drl, *_ = wb.do_wb_interp(
        np.float32(aws_t), aws_u_ts,
        cn_ts=np.full(2, 80.0, dtype="float32"),
        pr_ts=zeros, et_ts=zeros, eto_ts=np.ones(2, dtype="float32"),
        init_dru_frac=dr_u_prev / aws_u_prev,
        init_drl_frac=dr_l_prev / (aws_t - aws_u_prev),
        crop_switch_rule=rule,
    )
    return float(dru[1]), float(drl[1])


def _column_water(dr_u, dr_l, aws_u, aws_t):
    return (aws_u - dr_u) + (aws_t - aws_u - dr_l)


# (dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t). Both cases have a
# vertical contrast (the two reservoirs start at different fractional
# depletions).
EXPAND = (120.0, 50.0, 150.0, 250.0, 400.0)    # upper grows, lower cedes a slice
CONTRACT = (200.0, 30.0, 250.0, 120.0, 400.0)  # upper shrinks, upper cedes a slice


@pytest.mark.parametrize("case", [EXPAND, CONTRACT])
@pytest.mark.parametrize("rule", ["uniform", "boundary_transfer"])
def test_conserves_column_water(case, rule):
    dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t = case
    dr_u_new, dr_l_new = _switch(*case, rule)

    before = _column_water(dr_u_prev, dr_l_prev, aws_u_prev, aws_t)
    after = _column_water(dr_u_new, dr_l_new, aws_u_new, aws_t)
    npt.assert_allclose(after, before, atol=1e-2)


@pytest.mark.parametrize("case,expands", [(EXPAND, True), (CONTRACT, False)])
def test_boundary_transfer_preserves_source_reservoir_depletion_fraction(case, expands):
    # The reservoir that cedes capacity keeps its fractional
    # depletion. On expand the lower cedes. On contract the upper cedes.
    dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t = case
    dr_u_b, dr_l_b = _switch(*case, "boundary_transfer")

    if expands:  # lower is the source reservoir
        aws_src_prev, aws_src_new = aws_t - aws_u_prev, aws_t - aws_u_new
        dr_src_prev, dr_src_b = dr_l_prev, dr_l_b
    else:        # upper is the source reservoir
        aws_src_prev, aws_src_new = aws_u_prev, aws_u_new
        dr_src_prev, dr_src_b = dr_u_prev, dr_u_b

    frac_prev = dr_src_prev / aws_src_prev
    npt.assert_allclose(dr_src_b / aws_src_new, frac_prev, atol=1e-4)


def test_rules_agree_on_uniform_column():
    # If both reservoirs start at the same fractional depletion there is no
    # vertical contrast to redistribute, and the two rules give the same result.
    aws_t, aws_u_prev, aws_u_new = 400.0, 150.0, 250.0
    frac = 0.5  # dr = frac * aws in both reservoirs
    dr_u_prev, dr_l_prev = frac * aws_u_prev, frac * (aws_t - aws_u_prev)

    b = _switch(dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t,
                "boundary_transfer")
    u = _switch(dr_u_prev, dr_l_prev, aws_u_prev, aws_u_new, aws_t, "uniform")
    npt.assert_allclose(b, u, atol=1e-2)


def test_no_switch_is_identical_across_rules():
    # With an unchanged upper capacity no redistribution is triggered, so the
    # two rules must produce identical output (default-off guarantee).
    b = _switch(60.0, 100.0, 150.0, 150.0, 400.0, "boundary_transfer")
    u = _switch(60.0, 100.0, 150.0, 150.0, 400.0, "uniform")
    npt.assert_array_equal(b, u)
