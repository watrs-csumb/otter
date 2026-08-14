import numpy as np
from numba import njit, prange

@njit
def etof_interp(et_ts, eto_ts, nodata=-9999., dtype='float32', max_etof=2.0, min_eto=0.01):
    etof_ts = np.zeros_like(eto_ts, dtype=dtype)

    # fill leading empty values with first EToF
    first_et_ind = np.argmax(et_ts != nodata)
    first_et_val = et_ts[first_et_ind]
    first_etof = first_et_val / eto_ts[first_et_ind]
    etof_ts[:first_et_ind+1] = min(first_etof, max_etof)

    # handle zero eto values
    #eto_ts[eto_ts==0] = eto_ts[eto_ts!=0].min()
    eto_ts[eto_ts < min_eto] = min_eto 

    start_ind = first_et_ind
    for i in range(first_et_ind+1, et_ts.size):
        # find next non-missing et value
        if et_ts[i] != nodata:
            start_etof = min(et_ts[start_ind] / eto_ts[start_ind], max_etof)
            end_etof = min(et_ts[i] / eto_ts[i], max_etof)

            # linear interpolate from start to end
            etof_ts[start_ind:i+1] = np.linspace(start_etof,
                                                 end_etof,
                                                 i+1-start_ind)

            # set current index as start_ind
            start_ind = i
        # Reached missing value
        else:
            # end of et_ts
            if i == et_ts.size-1:
                etof_ts[start_ind:] = et_ts[start_ind] / eto_ts[start_ind]
            # not end of et_ts
            else:
                continue

    # return interpolated ET
    return etof_ts * eto_ts

# all args are arrays with length of ts
# aws and cn vals are just repeated until end of year
@njit
def do_wb_interp(aws_max: float, aws_u_ts: np.ndarray,
                 cn_ts: np.ndarray, pr_ts: np.ndarray,
                 et_ts: np.ndarray, eto_ts: np.ndarray,
                 nodata=-9999., init_dru_frac=1.,
                 init_drl_frac=1., mad_frac=1.,
                 crop_switch_rule="uniform"):
    if init_dru_frac > mad_frac:
        raise Exception("init_dru_frac is larger than mad_frac (starting more depleted than allowed)")

    if crop_switch_rule != "uniform" and crop_switch_rule != "boundary_transfer":
        raise Exception("crop_switch_rule must be 'uniform' or 'boundary_transfer'")

    num_steps = et_ts.size

    # start empty
    last_dru = aws_u_ts[0]*init_dru_frac
    max_drl = aws_max - aws_u_ts[0]
    last_drl = max_drl*init_drl_frac
    last_aws_u = aws_u_ts[0]

    dru_ts = np.zeros(num_steps, dtype='float32')
    drl_ts = np.zeros(num_steps, dtype='float32')
    perc_ts = np.zeros(num_steps, dtype='float32')
    dperc_ts = np.zeros(num_steps, dtype='float32')
    ro_ts = np.zeros(num_steps, dtype='float32')
    etaw_ts = np.zeros(num_steps, dtype='float32')
    peff_ts = np.zeros(num_steps, dtype='float32')

    # not currently handling locations with no data
    # TODO something less dumb than this
    if sum(et_ts != nodata) == 0:
        return (dru_ts, drl_ts, perc_ts, dperc_ts, ro_ts,
                etaw_ts, peff_ts, et_ts)
    else:
        et_ts = etof_interp(et_ts, eto_ts, nodata=nodata)

    for i in range(num_steps):
        pr = pr_ts[i]
        et = et_ts[i]

        if aws_u_ts[i] != last_aws_u:
            # Crop switch: the upper reservoir capacity changed with the new
            # crop's max rooting depth. Re-split the carried-over depletion
            # between the two reservoirs. The total column capacity (aws_max)
            # and the total column water content are both conserved.
            if crop_switch_rule == "uniform":
                # paper Eqns 10-11: distribute total depletion in proportion to
                # the new capacities, i.e. reset the column to mean water content.
                total_dep = last_dru + last_drl
                last_dru = total_dep * aws_u_ts[i] / aws_max
                last_drl = total_dep * (aws_max - aws_u_ts[i]) / aws_max
            else:
                # boundary_transfer: move only the depth band that changes
                # ownership. Its exact capacity is d_aws_u = aws_u_ts[i] - old
                # upper capacity, and it carries water at the source reservoir's
                # fractional water content. Works in water content space,
                # w = aws - dr, and recovers depletion at the end.
                last_aws_l = aws_max - last_aws_u
                new_aws_l = aws_max - aws_u_ts[i]
                last_w_u = last_aws_u - last_dru
                last_w_l = last_aws_l - last_drl
                d_aws_u = aws_u_ts[i] - last_aws_u
                if d_aws_u > 0:
                    # upper expands into lower, at lower's water content
                    # (guard last_aws_l == 0 is logically unreachable here)
                    if last_aws_l == 0:
                        new_w_u = last_w_u
                    else:
                        new_w_u = last_w_u + d_aws_u * last_w_l / last_aws_l
                else:
                    # upper contracts, ceding to lower at upper's water content
                    # (guard last_aws_u == 0 is logically unreachable here)
                    if last_aws_u == 0:
                        new_w_u = last_w_u
                    else:
                        new_w_u = last_w_u + d_aws_u * last_w_u / last_aws_u
                new_w_l = (last_w_u + last_w_l) - new_w_u
                # defensive clamp. Already holds for 0 <= MRD <= 2.4 m
                last_dru = min(max(aws_u_ts[i] - new_w_u, 0.0), aws_u_ts[i])
                last_drl = min(max(new_aws_l - new_w_l, 0.0), new_aws_l)
            last_aws_u = aws_u_ts[i]

            # temporary small fudge
            if last_dru / last_aws_u - 0.01 > mad_frac:
                raise Exception("depletion exceed max allowable depletion after crop switch")

        S = (25400-254*cn_ts[i])/cn_ts[i]
        if pr > 0.2*S:
            ro = (pr-0.2*S)**2/(pr+0.8*S)
        else:
            ro = 0

        perc = max(pr - et - ro - last_dru, 0)
        dru = min(max(last_dru - pr + et + ro, 0),
                      mad_frac*aws_u_ts[i])

        dperc = max(perc - last_drl, 0)
        drl = min(max(last_drl - perc, 0),
                  aws_max - aws_u_ts[i])

        pr_eff = pr - dperc - ro
        dr_change = dru - last_dru
        etaw = et + perc - dperc - pr_eff - dr_change
        #if etaw < -1e-4:
        #    import ipdb
        #    ipdb.set_trace()

        last_dru = dru
        last_drl = drl

        dru_ts[i] = dru
        drl_ts[i] = drl
        perc_ts[i] = perc
        dperc_ts[i] = dperc
        ro_ts[i] = ro
        etaw_ts[i] = etaw
        peff_ts[i] = pr_eff

    return (dru_ts, drl_ts, perc_ts, dperc_ts, ro_ts, etaw_ts,
            peff_ts, et_ts)

def run_df(df, nodata=-9999, init_dru_frac=1., init_drl_frac=1., mad_frac=1.,
           crop_switch_rule="uniform"):
    missing = [x for x in ["pr", "et", "eto", "aws_u", "aws_max", "cn"] if x not in df.columns]
    if any(missing):
        raise Exception(f"input DataFrame missing required column(s) {missing}")

    def n(attr):
        return df[attr].to_numpy(dtype="float32")

    dru, drl, perc, dperc, ro, etaw, peff, et\
        = do_wb_interp(n("aws_max")[0], n("aws_u"), n("cn"), n("pr"),
                       n("et"), n("eto"),
                       init_dru_frac=init_dru_frac,
                       init_drl_frac=init_drl_frac,
                       nodata=nodata, mad_frac=mad_frac,
                       crop_switch_rule=crop_switch_rule)

    df["dru"] = dru
    df["drl"] = drl
    df["perc"] = perc
    df["dperc"] = dperc
    df["ro"] = ro
    df["etaw"] = etaw
    df["peff"] = peff
    df["et_interp"] = et

    return df
