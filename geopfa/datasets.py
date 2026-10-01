"""Functions to fetch sample data for geoPFA."""

from pathlib import Path
import shutil
from typing import Literal

import numpy as np

from geopfa.io.data_readers import GeospatialDataReaders, safe_json_load
from geopfa.io.data_writers import GeospatialDataWriters
from geopfa.processing import Cleaners, Processing

import pooch
from pooch.processors import Unzip


dogbert_newberry = pooch.create(
    path=pooch.os_cache("geoPFA"),
    base_url="https://github.com/NatLabRockies/geoPFA/releases/download/{version}/",
    version="v0.0.20",
    registry={
        "newberry_tutorial_data.zip": "sha256:d9168678b1e63f52a2e73b18d8b26e93ac502b3025e197391ccb26f3081f6c4c",
    },
)


def setup_newberry_tutorial_data(target_dir: Path) -> None:
    """
    Download and extract the Newberry tutorial dataset into a target directory.

    The contents of the zip are copied into `target_dir` exactly as structured.
    """
    filename = "newberry_tutorial_data.zip"

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = dogbert_newberry.fetch(filename, processor=Unzip())

    if not extracted_files:
        raise RuntimeError(f"No files extracted from {filename}")

    extracted_paths = [Path(p).resolve() for p in extracted_files]

    # Find the common extraction root of all extracted files
    common_root = Path(
        __import__("os").path.commonpath([str(p) for p in extracted_paths])
    )

    # If the zip contains a single top-level folder (for example "data/"),
    # unwrap that folder so we do not create data/data/.
    top_level_items = list(common_root.iterdir())
    if len(top_level_items) == 1 and top_level_items[0].is_dir():
        source_dir = top_level_items[0]
    else:
        source_dir = common_root

    # Copy the full directory structure into target_dir
    for item in source_dir.iterdir():
        dest = target_dir / item.name

        if dest.exists():
            continue

        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Data extracted to: {target_dir}")


NewberryDimensions = Literal["2d", "3d"]


def _newberry_processed_inputs_exist(
    config_path: Path, data_dir: Path
) -> bool:
    """Return whether a Newberry example has both processed inputs."""
    return config_path.is_file() and any(data_dir.rglob("*_processed.csv"))


def ensure_newberry_tutorial_processed_data(
    project_dir: Path, *, dimensions: NewberryDimensions
) -> tuple[Path, Path]:
    """Populate one Newberry example's raw and processed tutorial inputs.

    The released tutorial archive is downloaded into ``project_dir / "data"``
    by :func:`setup_newberry_tutorial_data`, exactly where the established
    Newberry notebooks expect it.  The first call also builds the ignored
    ``*_processed.csv`` layers using the non-plotting operations from the
    corresponding Newberry preprocessing notebook.  Later calls reuse those
    processed inputs unchanged.
    """
    if dimensions not in {"2d", "3d"}:
        raise ValueError("dimensions must be either '2d' or '3d'")

    project_dir = Path(project_dir).expanduser().resolve()
    data_dir = project_dir / "data"
    config_path = (
        project_dir / "config" / "newberry_superhot_processed_config.json"
    )
    setup_newberry_tutorial_data(data_dir)

    if not _newberry_processed_inputs_exist(config_path, data_dir):
        _prepare_newberry_processed_data(project_dir, dimensions)

    if not _newberry_processed_inputs_exist(config_path, data_dir):
        raise RuntimeError(
            "Newberry tutorial preprocessing completed without writing the "
            "required processed configuration and layer files"
        )
    return config_path, data_dir


def _prepare_newberry_processed_data(
    project_dir: Path, dimensions: NewberryDimensions
) -> None:
    """Create Newberry processed layers using the established tutorial recipe."""
    config_dir = project_dir / "config"
    data_dir = project_dir / "data"
    source_config = config_dir / "newberry_superhot_config.json"
    if not source_config.is_file():
        raise FileNotFoundError(
            f"Newberry tutorial configuration not found: {source_config}"
        )

    pfa = GeospatialDataReaders.gather_data(
        data_dir,
        safe_json_load(source_config),
        [".csv", ".shp"],
        validate=dimensions == "2d",
    )
    if dimensions == "2d":
        _prepare_newberry_2d(
            pfa,
            clean=Cleaners,
            processing=Processing,
            writer=GeospatialDataWriters,
            config_dir=config_dir,
            data_dir=data_dir,
        )
    else:
        _prepare_newberry_3d(
            pfa,
            clean=Cleaners,
            processing=Processing,
            writer=GeospatialDataWriters,
            config_dir=config_dir,
            data_dir=data_dir,
        )


def _prepare_newberry_2d(  # noqa: PLR0913
    pfa: dict,
    *,
    clean: object,
    processing: object,
    writer: object,
    config_dir: Path,
    data_dir: Path,
) -> None:
    """Apply the 2-D Newberry preprocessing notebook's data operations."""
    pfa = clean.set_crs(pfa, target_crs=26910)
    criteria = "geologic"
    extent_layer = pfa["criteria"][criteria]["components"]["heat"]["layers"][
        "mt_resistivity_joint_inv"
    ]["data"]
    extent = clean.get_extent(extent_layer, dim=2)

    flattened_layers: dict[str, dict] = {}
    for component, component_data in pfa["criteria"][criteria][
        "components"
    ].items():
        for layer, layer_config in component_data["layers"].items():
            if (
                not layer_config.get("is_3d")
                or str(layer_config.get("needs_flattening", "yes")).lower()
                == "no"
            ):
                continue
            if layer in flattened_layers:
                pfa["criteria"][criteria]["components"][component]["layers"][
                    layer
                ] = flattened_layers[layer].copy()
                continue
            pfa = processing.convert_3d_to_2d(
                pfa, criteria, component=component, layer=layer
            )
            flattened_layers[layer] = pfa["criteria"][criteria]["components"][
                component
            ]["layers"][layer].copy()

    interpolated_layers: dict[str, dict] = {}
    for criterion, criterion_data in pfa["criteria"].items():
        for component, component_data in criterion_data["components"].items():
            for layer, layer_config in component_data["layers"].items():
                if layer_config.get("processing_method") != "interpolate":
                    continue
                if layer in interpolated_layers:
                    cached = interpolated_layers[layer]
                    destination = pfa["criteria"][criterion]["components"][
                        component
                    ]["layers"][layer]
                    destination["model"] = cached["model"].copy()
                    destination["model_data_col"] = cached["model_data_col"]
                    destination["model_units"] = cached["model_units"]
                    continue
                pfa = processing.interpolate_points(
                    pfa,
                    criteria=criterion,
                    component=component,
                    layer=layer,
                    interp_method="linear",
                    nx=300,
                    ny=300,
                    extent=extent,
                )
                interpolated_layers[layer] = pfa["criteria"][criterion][
                    "components"
                ][component]["layers"][layer].copy()

    for component in ("heat", "reservoir", "insulation"):
        pfa = processing.weighted_distance_from_points(
            pfa,
            criteria=criteria,
            component=component,
            layer="earthquakes",
            extent=extent,
            nx=300,
            ny=300,
            alpha=1000,
        )
        layer_data = pfa["criteria"][criteria]["components"][component][
            "layers"
        ]["earthquakes"]
        layer_data["model"] = clean.filter_geodataframe(
            layer_data["model"], layer_data["model_data_col"], quantile=0.9
        )

    pfa = processing.process_faults(
        pfa,
        criteria=criteria,
        component="reservoir",
        layer="lineaments",
        extent=extent,
        nx=300,
        ny=300,
        alpha_fault=5500.0,
        alpha_intersection=3500.0,
        weight_fault=0.7,
        weight_intersection=0.3,
        use_intersections=True,
    )
    writer.save_processed_layers(pfa, data_dir)
    writer.save_clean_pfa_config(
        pfa, config_dir / "newberry_superhot_processed_config.json"
    )


def _prepare_newberry_3d(  # noqa: PLR0912, PLR0913
    pfa: dict,
    *,
    clean: object,
    processing: object,
    writer: object,
    config_dir: Path,
    data_dir: Path,
) -> None:
    """Apply the 3-D Newberry preprocessing notebook's data operations."""
    pfa = clean.set_crs(pfa, target_crs=26910)
    criteria = "geologic"
    target_z_meas = "m-msl"
    extent_layer = pfa["criteria"][criteria]["components"]["heat"]["layers"][
        "mt_resistivity_joint_inv"
    ]["data"]
    extent = clean.get_extent(extent_layer)
    extent[2] = np.float64(-6000)

    pfa = processing.extrude_2d_to_3d(
        pfa,
        criteria=criteria,
        component="reservoir",
        layer="lineaments",
        extent=extent,
        target_z_meas=target_z_meas,
        nz=50,
    )
    for criterion_data in pfa["criteria"].values():
        for component_data in criterion_data["components"].values():
            for layer_data in component_data["layers"].values():
                if target_z_meas != layer_data["z_meas"]:
                    layer_data["data"] = clean.convert_z_measurements(
                        layer_data["data"], layer_data["z_meas"], target_z_meas
                    )

    for component in ("heat", "reservoir"):
        layer_data = pfa["criteria"][criteria]["components"][component][
            "layers"
        ]["mt_resistivity_joint_inv"]
        layer_data["data"] = clean.filter_geodataframe(
            layer_data["data"], layer_data["data_col"], quantile=0.65
        )
    for component in ("heat", "reservoir", "insulation"):
        layer_data = pfa["criteria"][criteria]["components"][component][
            "layers"
        ]["mt_resistivity_joint_inv"]
        data = layer_data["data"].copy()
        column = layer_data["data_col"]
        data.loc[data[column] == 0, column] = 0.00001
        data[column] = np.log(data[column])
        values = data[column]
        data = data[
            values.notna()
            & values.replace(
                [float("inf"), float("-inf")], float("nan")
            ).notna()
        ]
        layer_data["data"] = data
        layer_data["units"] = "log(ohm-m)"

    interpolated_layers: dict[str, dict] = {}
    for criterion, criterion_data in pfa["criteria"].items():
        for component, component_data in criterion_data["components"].items():
            for layer, layer_config in component_data["layers"].items():
                if layer_config.get("processing_method") != "interpolate":
                    continue
                if layer in interpolated_layers:
                    cached = interpolated_layers[layer]
                    destination = pfa["criteria"][criterion]["components"][
                        component
                    ]["layers"][layer]
                    destination["model"] = cached["model"].copy()
                    destination["model_data_col"] = cached["model_data_col"]
                    destination["model_units"] = cached["model_units"]
                    continue
                pfa = processing.fast_interpolate_points_3d(
                    pfa,
                    criteria=criterion,
                    component=component,
                    layer=layer,
                    nx=100,
                    ny=100,
                    nz=50,
                    extent=extent,
                    method="nearest",
                )
                interpolated_layers[layer] = pfa["criteria"][criterion][
                    "components"
                ][component]["layers"][layer].copy()

    for component in ("heat", "reservoir", "insulation"):
        pfa = processing.weighted_distance_from_points_3d(
            pfa,
            criteria=criteria,
            component=component,
            layer="earthquakes",
            extent=extent,
            nx=100,
            ny=100,
            nz=50,
            alpha=5000,
        )
        layer_data = pfa["criteria"][criteria]["components"][component][
            "layers"
        ]["earthquakes"]
        layer_data["model"] = clean.filter_geodataframe(
            layer_data["model"], layer_data["model_data_col"], quantile=0.99
        )

    layer_data = pfa["criteria"][criteria]["components"]["reservoir"][
        "layers"
    ]["ring_faults"]
    layer_data["data"] = processing.create_fault_surfaces_from_points(
        layer_data["data"], "Fault_Number"
    )
    for layer in ("ring_faults", "lineaments"):
        pfa = processing.distance_from_3d_solids(
            pfa,
            criteria=criteria,
            component="reservoir",
            layer=layer,
            extent=extent,
            nx=100,
            ny=100,
            nz=50,
        )
    writer.save_processed_layers(pfa, data_dir)
    writer.save_clean_pfa_config(
        pfa, config_dir / "newberry_superhot_processed_config.json"
    )
