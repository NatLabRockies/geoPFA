import numpy as np
import pytest

from geopfa import transformation
from geopfa.io.data_readers import GeospatialDataReaders


@pytest.mark.parametrize("method", ["None", "none", "NONE", None])
def test_transform_none_variants_are_noop(method):
    array = np.array([1.0, 2.0])

    result = transformation.transform(array, method)

    assert result is array


def test_transform_requires_method_argument():
    with pytest.raises(ValueError, match="must be specified"):
        transformation.transform(np.array([1.0]))


@pytest.mark.parametrize("method", ["", "  "])
def test_transform_rejects_blank_method(method):
    with pytest.raises(ValueError, match="supported method or 'none'"):
        transformation.transform(np.array([1.0]), method)


@pytest.mark.parametrize(
    "gather_method", ["gather_data", "gather_processed_data"]
)
def test_gather_warns_when_transformation_method_is_missing(
    tmp_path, capsys, gather_method
):
    component_dir = tmp_path / "criterion" / "component"
    component_dir.mkdir(parents=True)
    pfa = {
        "criteria": {
            "criterion": {
                "components": {"component": {"layers": {"layer": {}}}}
            }
        }
    }

    if gather_method == "gather_data":
        GeospatialDataReaders.gather_data(tmp_path, pfa, file_types=[])
    else:
        GeospatialDataReaders.gather_processed_data(tmp_path, pfa, crs=None)

    output = capsys.readouterr().out
    assert "Warning: no transformation method specified" in output
    assert "criterion/component/layer" in output
