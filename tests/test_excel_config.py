import json

import pandas as pd
import pytest

from geopfa.io.data_readers import GeospatialDataReaders


pytest.importorskip("openpyxl")


def test_excel_to_pfa_json(tmp_path):
    rows = [
        {
            "criteria": "geologic",
            "criteria_weight": 1,
            "component": "heat",
            "component_weight": 0.6,
            "pr0": 0.5,
            "layer": "temperature",
            "layer_weight": 1,
            "units": "degC",
            "data_col": "temperature",
            "x_col": "x",
            "y_col": "y",
            "transformation_method": "none",
            "processing_method": "interpolate",
        },
        {
            "criteria": "",
            "criteria_weight": "",
            "component": "structure",
            "component_weight": 0.4,
            "pr0": 0.5,
            "layer": "faults",
            "layer_weight": 1,
            "units": "none",
            "data_col": "NONE",
            "x_col": "",
            "y_col": "",
            "transformation_method": "negate",
            "processing_method": "distance",
        },
    ]
    excel_path = tmp_path / "config.xlsx"
    json_path = tmp_path / "config.json"
    pd.DataFrame(rows).to_excel(excel_path, index=False)

    pfa = GeospatialDataReaders.excel_to_pfa_json(excel_path, json_path)

    faults = pfa["criteria"]["geologic"]["components"]["structure"]["layers"][
        "faults"
    ]
    assert faults["data_col"] is None
    assert "x_col" not in faults
    assert faults["weight"] == 1.0
    assert json.loads(json_path.read_text(encoding="utf-8")) == pfa


def test_excel_to_pfa_json_reports_missing_columns(tmp_path):
    excel_path = tmp_path / "config.xlsx"
    pd.DataFrame({"criteria": ["geologic"]}).to_excel(excel_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        GeospatialDataReaders.excel_to_pfa_json(excel_path)
