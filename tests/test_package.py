"""The package's public surface: what a user imports, and what a plain install refuses clearly.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""


# ----------------------------------------------------------------------------- taskcard, audited 2026-09-22
def _cards():
    import json, os
    return {c["name"]: c for c in json.load(open(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "taskcards.json")))}


def test_task_cards_count_stacked_time_steps_not_layers():
    """The cards counted layers, so models that stack 12 monthly scenes in one layer read as 1 time step."""
    c = _cards()
    for name, n in (("awf", 12), ("nandi", 12), ("mozambique_lulc", 8), ("togo_cropland", 8), ("fields_of_the_world", 8)):
        assert c[name]["inputs"]["sentinel2_l2a"]["n_timesteps"] == n, name


def test_task_cards_read_the_published_legend_the_split_grid_and_every_model():
    c = _cards()
    assert len(c["ecosystem_type_mapping"]["task"]["classes"]) == 60 and c["ecosystem_type_mapping"]["windows"]["grid_size"] == 10.0
    assert {"kenya_lulc_croptype:cropland", "kenya_lulc_croptype:maize"} <= set(c) and "kenya_lulc_croptype" not in c
    assert c["mangrove"]["inputs"] and c["satlas_solar_farm"]["inputs"]                    # DataInput specs are read
    assert c["mangrove"]["audit"]["output_is_dense"] is False                              # a pooling decoder is block-constant
    assert "decoder" not in c["forest_loss_driver"]["outputs"]                             # a classification head pools by design
    assert any("emits 3 channels" in w for w in c["togo_cropland"]["warnings"])
