"""Temporal test subsets used by the supplementary analysis."""
import numpy as np

ERA = "./data/eval/clinvar_pb_2026_split/precirc_era_annotation.npz"

def _era_masks(lab):
    z = np.load(ERA, allow_pickle=True)
    p18, t18, s18 = z["present_2018"], z["tier_2018"], z["star_2018"]
    p21, t21, s21 = z["present_2021"], z["tier_2021"], z["star_2021"]
    dmc = z["date_max_conc"]
    return {
        "all_era": np.ones(len(lab), bool),
        "pre_spliceai_release": p18 & (t18 == lab),
        "pre_spliceai_release_2star": p18 & (t18 == lab) & (s18 >= 2),
        "pre_spliceai_subdate": (dmc > 0) & (dmc < 20190117),
        "pre_pangolin_release": p21 & (t21 == lab),
        "pre_pangolin_release_2star": p21 & (t21 == lab) & (s21 >= 2),
        "pre_pangolin_subdate": (dmc > 0) & (dmc < 20210706),
    }
