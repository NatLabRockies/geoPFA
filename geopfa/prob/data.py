"""Generic raster sampling utilities for probabilistic geoPFA workflows."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


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


__all__ = ["sample_evidence_at_wells"]
