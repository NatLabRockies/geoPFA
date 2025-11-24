"""
Set of various methods to weight and combine data layers for use in PFA.
The methods included in this class are based on those outlined by the PFA
Best Practices Report (Pauling et al. 2023).
"""

import numpy as np
from .transformation import VoterVetoTransformation
from geopfa.layer_combination import get_w0, WeightsOfEvidence


class VoterVeto:
    """Class of functions to weight and combine data layers using the voter-veto method.
    This method is based on a generalized linear model and is defined as a best practice
    in the PFA Best Practices Report (Pauling et al. 2023)."""

    @staticmethod
    def get_w0(Pr0):
        return get_w0(Pr0)

    @staticmethod
    def voter(w, z, w0):
        """
        Combine processed, transformed, and scaled 2D data layers into a 'favorability'
        grid for a specific required resource component using a generalized linear model.

        Parameters
        ----------
        w : ndarray
            Array of weights of shape (n,1), where n is the number of input data layers,
            sorted in order of the input data layers.
        z : np.array
            Array containing processed, transformed, and scaled 2D data layers rasterized in
            np.arrays - all of which should be on the same grid. Shape (m,x,y), where m is
            the number of data layers
        w0 : float
            Value used to incorporate a reference 'favorability' (prior 'favorability') into
            the voter equation (generalized linear model). Is specific to a required
            component of a resource.
        Returns
        -------
        PrX : np.array
            Rasterized array of 'favorabilities' for an individual required resource component being
            present (i.e., heat, fluid, perm, etc.). Shape (x,y)
        """
        # Reshape w to be broadcastable with z
        w = w.reshape(-1, 1, 1)
        e = -w0 - np.nansum(w*z,axis=0)
        PrX = 1/(1 + np.exp(e))
        return PrX

    @staticmethod
    def veto(PrXs):
        """
        Combine component 'favorability' grids into a resource 'favorability' model, vetoing
        areas where any one component is not present (0% 'favorability').This method
        combines component 'favorability' grids by element-wise multiplication.

        Parameters
        ----------
        PrXs : np.array
            Array of rasterized 'favorability' arrays for each required component of a resource

        Returns
        -------
        PrR : np.array
            Array of rasterized 'favorability' arrays of a resource being present, taking into
            account all components (i.e., heat, fluid, perm, etc.). Shape (x,y)
        """
        # TODO: Incorporate component/criteria weights!!
        PrR = np.zeros(PrXs[0].shape)
        for i, c in enumerate(PrXs):
            if i == 0:
                PrR = c
            else:
                PrR = np.multiply(PrR, c)
        return PrR

    @staticmethod
    def modified_veto(PrXs,w,veto=True):
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
            account all components (i.e., heat, fluid, perm, etc.). Shape (x,y)
        """
        PrR = np.zeros(PrXs[0].shape)
        max_PrR = 1

        for i, c in enumerate(PrXs):
            if i == 0:
                max_PrR = c
            else:
                max_PrR = np.multiply(max_PrR, c)
            wPrX = w[i] * c
            PrR += wPrX
            if veto is True:
                PrR[c == 0] = 0

        # Normalize and scale to maintain valid 'favorability' distribution
        PrR = PrR / np.max(PrR) * np.max(max_PrR)
        return PrR

    @classmethod
    def do_voter_veto(cls, pfa, normalize_method, component_veto=False, criteria_veto=True, normalize=True,norm_to=5):
        """
        Combine individual data layers into a resource 'favorability' model,
        vetoing areas where any one component is not present (0% 'favorability').
        This method is described in detail in Ito et al., 2017 from the Hawaii
        Geothermal PFA project.

        Parameters
        ----------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames to weight and combine into 'favorability'
            models.
        normalize_method : str
            Mathod to use to normalize data layers. Can be one of ['minmax','mad']
        Returns
        -------
        pfa : dict
            Config specifying criteria, components, and data layers' relationship to one another.
            Includes data layers' associated GeoDataFrames of newly produced 'favorability' models.
        """
        PrRs = []; w_criteria = []
        for criteria in pfa['criteria']:
            PrXs = []; w_components = []
            for component in pfa['criteria'][criteria]['components']:
                z = []; w_layers = []
                Pr0 = pfa['criteria'][criteria]['components'][component]['pr0']
                w0 = cls.get_w0(Pr0)
                for layer in pfa['criteria'][criteria]['components'][component]['layers']:
                    print(layer)
                    model = pfa['criteria'][criteria]['components'][component]['layers'][layer]['model']
                    col = pfa['criteria'][criteria]['components'][component]['layers'][layer]['model_data_col']
                    transformation_method = pfa['criteria'][criteria]['components'][component]['layers'][layer]['transformation_method']

                    model_array = VoterVetoTransformation.rasterize_model(gdf=model,col=col)
                    if transformation_method != "none":
                        model_array = VoterVetoTransformation.transform(model_array, transformation_method)
                    model_array = VoterVetoTransformation.normalize_array(model_array,method=normalize_method)
                    z.append(model_array)
                    w_layers.append(pfa['criteria'][criteria]['components'][component]['layers'][layer]['weight'])

                PrX = cls.voter(np.array(w_layers), np.array(z), w0)
                pfa['criteria'][criteria]['components'][component]['pr'] = VoterVetoTransformation.derasterize_model(PrX,gdf_geom=model)
                if normalize is True:
                    pfa['criteria'][criteria]['components'][component]['pr_norm'] = VoterVetoTransformation.normalize_gdf(
                        pfa['criteria'][criteria]['components'][component]['pr'],col='favorability',norm_to=norm_to)
                PrXs.append(PrX)
                w_components.append(pfa['criteria'][criteria]['components'][component]['weight'])

            PrR = cls.modified_veto(PrXs, np.array(w_components),veto=component_veto)
            pfa['criteria'][criteria]['pr'] = VoterVetoTransformation.derasterize_model(PrR,gdf_geom=model)
            if normalize is True:
                pfa['criteria'][criteria]['pr_norm'] = VoterVetoTransformation.normalize_gdf(
                    pfa['criteria'][criteria]['pr'],col='favorability',norm_to=norm_to)
            PrRs.append(PrR)
            w_criteria.append(pfa['criteria'][criteria]['weight'])

        PrR = cls.modified_veto(PrRs,np.array(w_criteria),veto=criteria_veto)
        pfa['pr'] = VoterVetoTransformation.derasterize_model(PrR,gdf_geom=model)
        if normalize is True:
            pfa['pr_norm'] = VoterVetoTransformation.normalize_gdf(pfa['pr'],col='favorability',norm_to=norm_to)
        return pfa
