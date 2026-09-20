"""Generic raster sampling utilities for probabilistic geoPFA workflows."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from geopfa.io.data_readers import GeospatialDataReaders, safe_json_load


def load_processed_pfa(
    config_path: str | Path,
    data_dir: str | Path,
    *,
    crs: str | int,
) -> tuple[dict, dict[str, Path]]:
    """Load a processed PFA and return every consumed file for provenance."""
    config_path = Path(config_path).resolve()
    data_dir = Path(data_dir).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"processed PFA config not found: {config_path}"
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"processed data directory not found: {data_dir}"
        )
    layers = sorted(data_dir.rglob("*_processed.csv"))
    if not layers:
        raise FileNotFoundError(
            f"no *_processed.csv layer files found below {data_dir}"
        )
    pfa = GeospatialDataReaders.gather_processed_data(
        data_dir,
        safe_json_load(config_path),
        crs=crs,
        validate=True,
        strict=True,
    )
    artifacts = {"processed_config": config_path}
    artifacts.update(
        {
            f"processed_data:{path.relative_to(data_dir).as_posix()}": path
            for path in layers
        }
    )
    return pfa, artifacts


def sample_evidence_at_wells(
    gdf: gpd.GeoDataFrame,
    raster_paths: dict[str, str | Path],
    band: int = 1,
    nodata_to_nan: bool = True,
) -> pd.DataFrame:
    """Sample named rasters at point locations in each raster's own CRS."""
    if gdf.crs is None:
        raise ValueError("gdf must have a valid CRS")
    if not raster_paths:
        raise ValueError("raster_paths cannot be empty")

    out = pd.DataFrame(index=gdf.index)
    for feature_name, raster_path in raster_paths.items():
        raster_path_obj = Path(raster_path)
        with rasterio.open(raster_path_obj) as dataset:
            if dataset.crs is None:
                raise ValueError(
                    f"evidence raster has no CRS: {raster_path_obj}"
                )
            points = gdf.to_crs(dataset.crs)
            coordinates = zip(
                points.geometry.x.to_numpy(),
                points.geometry.y.to_numpy(),
                strict=False,
            )
            values = np.fromiter(
                (
                    float(sample[0])
                    for sample in dataset.sample(coordinates, indexes=band)
                ),
                dtype=float,
                count=len(points),
            )
            if nodata_to_nan and dataset.nodata is not None:
                values[np.isclose(values, dataset.nodata)] = np.nan
            out[feature_name] = values
    return out


__all__ = ["load_processed_pfa", "sample_evidence_at_wells"]
