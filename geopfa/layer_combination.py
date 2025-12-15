"""
Set of various methods to weight and combine data layers for use in PFA.
The methods included in this class are based on those outlined by the PFA
Best Practices Report (Pauling et al. 2023).
"""

import warnings

import numpy as np

from .transformation import VoterVetoTransformation
from .transformation import normalize_gdf, normalize_array, detect_geom_dimension


def detect_pfa_dimension(pfa: dict) -> int:
    """Detect whether the PFA uses 2D or 3D geometries.

    All non-empty layer GeoDataFrames must share the same dimensionality.
    """
    dim = None

    for criteria in pfa.get("criteria", {}):
        for component in pfa["criteria"][criteria].get("components", {}):
            for layer_cfg in pfa["criteria"][criteria]["components"][component]['layers'].values():
                gdf = layer_cfg.get("model")
                if gdf is None or len(gdf) == 0:
                    continue

                layer_dim = detect_geom_dimension(gdf)

                if dim is None:
                    dim = layer_dim
                elif dim != layer_dim:
                    raise ValueError(
                        "PFA configuration mixes 2D and 3D geometries across layers. "
                        "All layers used in voter-veto must be consistently 2D or 3D."
                    )

    if dim is None:
        raise ValueError("Could not detect geometry dimension from PFA. "
                        "No non-empty layer models were found.")

    return dim


class WeightsOfEvidence:
    """Class of functions to weight and combine data layers using the weights of evidence
    method. This method examines multiple layers of evidence, calculates weights for each
    evidential layer based upon the spatial relationships of training points, which are
    located at known geothermal systems, and then produces a posterior 'favorability' raster
    surface and other related statistics. Weights of Evidence is defined as a best
    practice in the PFA Best Practices Report (Pauling et al. 2023)."""

    @classmethod
    def do_weights_of_evidence(cls):
        """
        Combine individual data layers into a resource 'favorability' model,
        using WoE.

        Parameters
        ----------
        pfa : dict

        Returns
        __________
        pfa : dict
        """
        print("NOT YET IMPLEMENTED")
        # # # Example below from: https://ishanjainoffical.medium.com/understanding-weight-of-evidence-woe-with-python-code-cd0df0e4001e
        # # # TODO: Enhance with this article: https://www.sciencedirect.com/science/article/pii/S0377027313002941?via%3Dihub
        # # # and Tularosa Basin reports/papers. Maybe Faulds work??
        # # Calculate WOE for Category 'A' and 'B'
        # category_counts = data['Category'].value_counts()
        # category_counts_pos = data[data['Target'] == 1]['Category'].value_counts()
        # category_counts_neg = data[data['Target'] == 0]['Category'].value_counts()

        # # Calculate WOE
        # woe_pos = np.log((category_counts_pos['A'] / category_counts['A']) / (category_counts_neg['A'] / category_counts['A']))
        # woe_neg = np.log((category_counts_pos['B'] / category_counts['B']) / (category_counts_neg['B'] / category_counts['B']))
        PrR = None
        return PrR


class VoterVeto:
    """Unified voter-veto driver that works for both 2D and 3D PFA configs.

    This keeps the current behavior of the legacy 2D and 3D implementations:
    - No special NaN treatment beyond whatever numpy does by default
    - No explicit grid-spacing or alignment checks
    The only new behavior here is:
    - Auto-detection of 2D vs 3D
    - A clear error if the PFA mixes 2D and 3D geometries
    """

    @staticmethod
    def get_w0(Pr0):
        """
        Derives w0 value from reference 'favorability', or prior 'favorability', using logit
        function.  Is specific to a required component of a resource.

        Parameters
        ----------
        Pr0 : float
            Reference 'favorability', or prior 'favorability'. Can be defined using expert
            opinion or other means. Is specific to a required component of a resource.

        Returns
        -------
        w0 : float
            Value used to incorporate a reference 'favorability' (prior 'favorability') into
            the voter equation (generalized linear model). Is specific to a required
            component of a resource.
        """

        w0 = np.log(Pr0 / (1 - Pr0))
        return w0

    @staticmethod
    def voter(w, z, w0):
        """
        Generalized voter for N-dimensional grids.

        Combine processed, transformed, and scaled data layers into a 'favorability'
        grid for a specific required resource component using a generalized linear model.

        Parameters
        ----------
        w : ndarray, shape (n_layers,)
            Array of weights, one per data layer.
        z : np.ndarray, shape (n_layers, *spatial_shape)
            Array containing processed, transformed, and scaled data layers.
        w0 : float
            Prior favorability term.

        Returns
        -------
        PrX : np.ndarray, shape spatial_shape
            Rasterized array of 'favorabilities' for an individual required resource component.
        """
        w = np.asarray(w)

        if w.ndim > 1:
            warnings.warn(
                "Weights array should be 1D, i.e. one value per data layer. "
                "Squeezing to 1D.",
                stacklevel=2,
            )
            w = w.squeeze()

        if z.ndim < 2:  # noqa: PLR2004
            raise ValueError(
                "z must have at least 2 dimensions: (n_layers, *spatial_dims)."
            )

        if w.shape[0] != z.shape[0]:
            raise ValueError(
                f"Number of weights ({w.shape[0]}) must match number of data layers ({z.shape[0]})."
            )

        # Broadcast weights over any number of spatial dimensions
        spatial_dims = (1,) * (z.ndim - 1)
        w_broadcast = w.reshape((w.shape[0], *spatial_dims))

        e = -w0 - np.nansum(w_broadcast * z, axis=0)
        PrX = 1 / (1 + np.exp(e))
        return PrX

    @staticmethod
    def veto(PrXs):
        """Original veto: element-wise multiplication of component favorabilities."""
        PrR = PrXs[0].copy()
        for c in PrXs[1:]:
            PrR = np.multiply(PrR, c)
        return PrR

    @staticmethod
    def modified_veto(PrXs, w, veto=True):
        """
        Combine component 'favorability' grids into a resource 'favorability' model, optionally
        vetoing areas where any one component is not present (0% 'favorability'). This method
        combines component 'favorability' grids using a weighted sum, and then normalizing.

        Parameters
        ----------
        PrXs : np.array
            Array of rasterized 'favorability' arrays for each required component or
            criteria of a resource
        w : np.array
            Array of weights for each component or criteria of a resource
        veto : boolean
            Boolean value indicating whether or not the function should set indices
            to zero where one component or criteria does not exist

        Returns
        -------
        PrR : np.array
            Array of rasterized 'favorability' arrays of a resource being present, taking into
            account all components (i.e., heat, fluid, perm, etc.).
        """
        PrR = np.zeros_like(PrXs[0])
        max_PrR = PrXs[0].copy()

        # Iterate from the second component onward
        for i, c in enumerate(PrXs):
            if i > 0:
                max_PrR = np.multiply(max_PrR, c)

            wPrX = w[i] * c
            PrR += wPrX

            if veto:
                PrR[c == 0] = 0

        # Normalize and scale to maintain valid 'favorability' distribution
        try:
            max_val = np.nanmax(PrR)
        except ValueError:
            # All values are NaN
            warnings.warn(
                "modified_veto: all values are NaN; normalization skipped.",
                stacklevel=2,
            )
            return PrR

        if max_val > 0:
            PrR = PrR / max_val * np.nanmax(max_PrR)
        else:
            warnings.warn(
                "modified_veto: maximum non-NaN of PrR is <= 0; normalization skipped.",
                stacklevel=2,
            )

        return PrR

    @staticmethod
    def prepare_for_combination(arr, nan_mode="default"):
        """
        Prepare a stack of layers/components for combination by handling NaNs.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (n_items, *spatial_shape), where n_items is the number
            of layers (at layer→component level), components (at component→criteria),
            or criteria (at criteria→final).
        nan_mode : {"default", "propagate_any"}
            Strategy for handling NaN values during aggregation:

            - "default":
                Propagates NaNs through cmobined layers only where *all* contributing
                inputs are NaN. NaNs present in only a subset of inputs are treated
                as neutral (non-contributing) evidence.

            - "propagate_any":
                Propagates NaNs through cmobined layers whenever *any* contributing
                input is NaN.

        Returns
        -------
        filled : np.ndarray
            Same shape as `arr`, with NaNs replaced according to `nan_mode`
            for the purpose of numerical combination.
        mask_nan : np.ndarray (bool)
            Boolean mask of shape `spatial_shape` indicating where the final
            combined result should be set to NaN.
        """
        arr = np.asarray(arr, dtype=float)

        if arr.ndim < 2:  # noqa: PLR2004
            raise ValueError(
                "Input to _prepare_for_combination must have at least 2 dimensions: "
                "(n_items, *spatial_dims)."
            )

        # mask_valid: True where that item has data at that pixel
        mask_valid = ~np.isnan(arr)

        # coverage masks
        has_data_any = np.any(mask_valid, axis=0)   # at least one item has data
        has_data_all = np.all(mask_valid, axis=0)   # all items have data

        # shared-no-data mask: no item has data here
        shared_nan_mask = ~has_data_any

        filled = arr.copy()

        if nan_mode == "default":
            for i in range(filled.shape[0]):
                layer = filled[i]
                # pixels where this layer is NaN but at least one other has data
                layer_nan = np.isnan(layer) & has_data_any
                if not np.any(layer_nan):
                    continue

                neutral = np.nanmedian(layer)
                if np.isnan(neutral):
                    # entire layer is NaN wherever there is coverage; fall back to 0
                    neutral = 0.0

                layer[layer_nan] = neutral
                filled[i] = layer

            mask_nan = shared_nan_mask

        elif nan_mode == "propagate_any":
            any_nan = ~np.all(mask_valid, axis=0)

            for i in range(filled.shape[0]):
                layer = filled[i]
                layer_nan = np.isnan(layer) & has_data_any
                if not np.any(layer_nan):
                    continue

                neutral = np.nanmedian(layer)
                if np.isnan(neutral):
                    neutral = 0.0

                layer[layer_nan] = neutral
                filled[i] = layer

            mask_nan = any_nan

        else:
            raise ValueError(
                f"Invalid nan_mode '{nan_mode}'. "
                "Must be one of {'default', 'propagate_any'}."
            )

        return filled, mask_nan

    @classmethod
    def do_voter_veto(  # noqa: PLR0915, PLR0914, PLR0913, PLR0917
        cls,
        pfa,
        normalize_method,
        component_veto=False,
        criteria_veto=True,
        normalize=True,
        norm_to=5,
        nan_mode="default",
    ):
        # TODO: refactor into helpers to satisy ignored Ruff hits
        """
        Combine individual data layers into a resource 'favorability' model,
        using the unified 2D/3D voter-veto implementation with NaN handling.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames to weight and combine into 'favorability'
            models.
        normalize_method : str
            Method to use to normalize data layers. Can be one of ['minmax','mad'].
        component_veto : bool
            Whether to apply veto at the component level.
        criteria_veto : bool
            Whether to apply veto at the criteria level.
        normalize : bool
            Whether to normalize output favorability GeoDataFrames.
        norm_to : float
            Max value for normalization of favorability in GeoDataFrames.
        nan_mode : {"default", "propagate_any"}
            Strategy for handling NaN values during aggregation:

            - "default":
                Propagates NaNs through cmobined layers only where *all* contributing
                inputs are NaN. NaNs present in only a subset of inputs are treated
                as neutral (non-contributing) evidence.

            - "propagate_any":
                Propagates NaNs through cmobined layers whenever *any* contributing
                input is NaN.

        Returns
        -------
        pfa : dict
            Updated PFA config with component, criteria, and final favorability models.
        """
        dim = detect_pfa_dimension(pfa)
        print(
            f"Combining {dim}D PFA layers with the voter-veto method. "
        )
        if nan_mode != "default":
            print(
                f"Nan mode: {nan_mode}."
            )

        if dim == 2:  # noqa: PLR2004
            rasterize = VoterVetoTransformation.rasterize_model_2d
            derasterize = VoterVetoTransformation.derasterize_model_2d
        else:
            rasterize = VoterVetoTransformation.rasterize_model_3d
            derasterize = VoterVetoTransformation.derasterize_model_3d

        PrRs = []
        w_criteria = []
        ref_shape = None
        last_model_geom = None

        for criteria in pfa["criteria"]:
            print(f"criterion: {criteria}")
            PrXs = []
            w_components = []

            for component in pfa["criteria"][criteria]["components"]:
                print(f"    component: {component}")
                z_layers = []
                w_layers = []
                Pr0 = pfa["criteria"][criteria]["components"][component]["pr0"]
                w0 = cls.get_w0(Pr0)

                for layer in pfa["criteria"][criteria]["components"][component]["layers"]:
                    print(f"        layer: {layer}")
                    layer_cfg = pfa["criteria"][criteria]["components"][component]["layers"][layer]
                    model = layer_cfg["model"]
                    col = layer_cfg["model_data_col"]
                    transformation_method = layer_cfg["transformation_method"]

                    last_model_geom = model.copy()  # keep reference for later derasterization

                    model_array = rasterize(model, col)

                    if ref_shape is None:
                        ref_shape = model_array.shape
                    elif model_array.shape != ref_shape:
                        raise ValueError("Layer grid shape mismatch.")

                    # transform
                    if transformation_method not in {"none", "None"}:
                        model_array = VoterVetoTransformation.transform(
                            model_array, transformation_method
                        )
                    print(f"        - Transformed with method: {transformation_method}")

                    # normalize
                    model_array = normalize_array(
                        model_array, method=normalize_method
                    )
                    print(f"        - Normalized with method: {normalize_method}")

                    z_layers.append(model_array)
                    w_layers.append(layer_cfg["weight"])

                z_arr = np.array(z_layers)
                w_arr = np.array(w_layers)

                # NaN handling at layer -> component level
                z_filled, mask_layers = cls.prepare_for_combination(
                    z_arr, nan_mode=nan_mode
                )

                # voter combination
                PrX = cls.voter(w_arr, z_filled, w0)
                # apply propagated NaNs
                PrX[mask_layers] = np.nan

                # derasterize to GeoDataFrame
                dr = derasterize(PrX, model)

                pfa["criteria"][criteria]["components"][component]["pr"] = dr

                if normalize:
                    pfa["criteria"][criteria]["components"][component]["pr_norm"] = (
                        normalize_gdf(
                            dr,
                            col="favorability",
                            norm_to=norm_to,
                        )
                    )

                PrXs.append(PrX)
                w_components.append(pfa["criteria"][criteria]["components"][component]["weight"])

            # NaN handling at component -> criteria level
            PrXs_filled, mask_components = cls.prepare_for_combination(
                np.array(PrXs), nan_mode=nan_mode
            )

            PrR_criteria = cls.modified_veto(
                PrXs_filled, np.array(w_components), veto=component_veto
            )
            PrR_criteria[mask_components] = np.nan

            # derasterize criteria-level favorability
            dr = derasterize(PrR_criteria, last_model_geom)

            pfa["criteria"][criteria]["pr"] = dr
            if normalize:
                pfa["criteria"][criteria]["pr_norm"] = normalize_gdf(
                    pfa["criteria"][criteria]["pr"],
                    col="favorability",
                    norm_to=norm_to,
                )

            PrRs.append(PrR_criteria)
            w_criteria.append(pfa["criteria"][criteria]["weight"])

        # NaN handling at criteria -> final level
        PrRs_filled, mask_criteria = cls.prepare_for_combination(
            np.array(PrRs), nan_mode=nan_mode
        )

        # final resource favorability
        PrR_final = cls.modified_veto(
            PrRs_filled, np.array(w_criteria), veto=criteria_veto
        )
        PrR_final[mask_criteria] = np.nan

        # derasterize final favorability
        dr = derasterize(PrR_final, last_model_geom)

        pfa["pr"] = dr
        if normalize:
            pfa["pr_norm"] = normalize_gdf(
                pfa["pr"], col="favorability", norm_to=norm_to
            )

        return pfa
