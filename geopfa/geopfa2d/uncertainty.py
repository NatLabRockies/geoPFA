"""Set of methods to model geospatial uncertainty in 2D"""
import numpy as np
import matplotlib.pyplot as plt
import geostatspy.geostats as geostats
import geostatspy.GSLIB as GSLIB
import matplotlib.colors as colors

class SGS:
    """
    A class of functions to perform Sequential Gaussian Simulation using geostatspy.
    """

    @staticmethod
    def get_cartesian1(lat=None, lon=None):
        """
        Converts latitude and longitude to Cartesian x-coordinate.

        Parameters:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            float: x-coordinate in Cartesian space.
        """
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        R = 6371  # Radius of the Earth in kilometers
        x = R * np.cos(lat) * np.cos(lon)
        return x

    @staticmethod
    def get_cartesian2(lat=None, lon=None):
        """
        Converts latitude and longitude to Cartesian y-coordinate.

        Parameters:
            lat (float): Latitude in degrees.
            lon (float): Longitude in degrees.

        Returns:
            float: y-coordinate in Cartesian space.
        """
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        R = 6371  # Radius of the Earth in kilometers
        y = R * np.cos(lat) * np.sin(lon)
        return y

    @staticmethod
    def get_vario(nug=0.0, nst=1, it1=1, cc1=1, azi1=0, hmaj1=30, hmin1=30):
        """
        Creates a variogram using GSLIB.

        Parameters:
            nug (float): Nugget effect.
            nst (int): Number of structures.
            it1 (int): Type of variogram structure.
            cc1 (float): Contribution of the first structure.
            azi1 (float): Azimuth angle for the first structure.
            hmaj1 (float): Major range for the first structure.
            hmin1 (float): Minor range for the first structure.

        Returns:
            dict: Variogram parameters.
        """
        varios = GSLIB.make_variogram(nug, nst, it1, cc1, azi1, hmaj1, hmin1)
        return varios

    @staticmethod
    def disc_data(df):
        """
        Adds Cartesian coordinates to a dataframe based on latitude and longitude.

        Parameters:
            df (pd.DataFrame): Dataframe containing LatDegree and LongDegree columns.

        Returns:
            pd.DataFrame: Dataframe with added 'X' and 'Y' columns for Cartesian coordinates.
        """
        a = df.apply(lambda x: SGS.get_cartesian1(x.LatDegree, x.LongDegree), axis=1)
        b = df.apply(lambda x: SGS.get_cartesian2(x.LatDegree, x.LongDegree), axis=1)
        df['X'] = a * -1
        df['Y'] = b * -1
        return df

    @staticmethod
    def loc_grid(array, xmin, xmax, ymin, ymax, vmin, vmax, df, xcol, ycol, vcol, title, xlabel, ylabel, vlabel, cmap):
        """
        Plots a grid and scatter overlay of geospatial data.

        Parameters:
            array (np.ndarray): 2D array of data values.
            xmin, xmax, ymin, ymax (float): Extents of the grid.
            vmin, vmax (float): Minimum and maximum values for color normalization.
            df (pd.DataFrame): Dataframe containing point data.
            xcol, ycol (str): Column names for x and y coordinates.
            vcol (str): Column name for value to color points by.
            title, xlabel, ylabel, vlabel (str): Labels for the plot and color bar.
            cmap (str): Colormap for visualization.
        """
        im = plt.imshow(array, cmap=cmap, extent=[xmin, xmax, ymin, ymax], norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        cbar = plt.colorbar(im, orientation="vertical", ticks=np.linspace(vmin, vmax, 10))
        cbar.set_label(vlabel, rotation=270, labelpad=20)
        plt.scatter(df[xcol], df[ycol], c=df[vcol], s=30, edgecolor='white', cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

    @staticmethod
    def local_expectation(realizations,nx,ny):
        """
        Calculates the local expectation (average) of realizations.

        Parameters:
            realizations (np.ndarray): 3D array of realizations (ny, nx, nreal).

        Returns:
            np.ndarray: 2D array of local expectations.
        """
        cond_exp = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                cond_exp[iy, ix] = np.average(realizations[iy, ix, :])
        return cond_exp

    @staticmethod
    def local_standard_deviation(realizations,nx,ny):
        """
        Calculates the local standard deviation of realizations.

        Parameters:
            realizations (np.ndarray): 3D array of realizations (ny, nx, nreal).

        Returns:
            np.ndarray: 2D array of local standard deviations.
        """
        local_stdev = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                local_stdev[iy, ix] = np.std(realizations[iy, ix, :])
        return local_stdev

    @staticmethod
    def local_percentile(realizations, p_value, nx, ny):
        """
        Calculates the local percentile of realizations.

        Parameters:
            realizations (np.ndarray): 3D array of realizations (ny, nx, nreal).
            p_value (float): Percentile value to compute.

        Returns:
            np.ndarray: 2D array of local percentiles.
        """
        percentile = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                percentile[iy, ix] = np.percentile(realizations[iy, ix, :], q=p_value)
        return percentile

    @staticmethod
    def local_probability_exceedance(realizations, threshold, nx, ny):
        """
        Calculates the local probability of exceeding a threshold.

        Parameters:
            realizations (np.ndarray): 3D array of realizations (ny, nx, nreal).
            threshold (float): Threshold value.

        Returns:
            np.ndarray: 2D array of probabilities of exceedance.
        """
        prob_exceed = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                prob_exceed[iy, ix] = np.sum(realizations[iy, ix, :] >= threshold) / realizations.shape[2]
        return prob_exceed

    def sim(df, vario, ktype, criteria_name,
        xcol, ycol, nx=200, ny = 20,
        xmn=2, ymn=2, ndmin=0, ndmax=20,nreal = 35):
        """
        Runs a sequential Gaussian simulation.

        Parameters:
            df (pd.DataFrame): Input dataframe with criteria data.
            vario (dict): Variogram parameters.
            ktype (int): Kriging type.
            criteria_name (str): Column name for simulation criteria.

        Returns:
            np.ndarray: 3D array of simulated realizations.
        """
        # sill = np.var(df[criteria_name].values)
        # skmean = np.average(df[criteria_name].values)
        x = df[xcol]; y = df[ycol]
        xsiz, ysiz = x.max() / nx, y.max() / nx
        realizations = np.zeros((ny, nx, nreal))

        for i in range(nreal):
            sim_grad = geostats.sgsim(
                df, 'X', 'Y', criteria_name, wcol=-1, scol=-1, tmin=-9999.9, tmax=9999.9, itrans=1,
                ismooth=0, dftrans=0, tcol=0, twtcol=0, zmin=df[criteria_name].min(),
                zmax=df[criteria_name].max(), ltail=2, ltpar=1, utail=2, utpar=df[criteria_name].max(),
                nsim=1, nx=nx, xmn=xmn, xsiz=xsiz, ny=ny, ymn=ymn, ysiz=ysiz, seed=73333 + i,
                ndmin=ndmin, ndmax=ndmax, nodmax=20, mults=0, nmult=2, noct=-1, ktype=ktype, colocorr=0.0, sec_map=0, vario=vario
            )
            realizations[:, :, i] = sim_grad
        return realizations

    def plot_stddev(realizations, df, new_df, criteria_name, nx, ny, cmap='viridis'):
        """
        Plots the standard deviation of realizations.

        Parameters:
            realizations (np.ndarray): 3D array of realizations (ny, nx, nreal).
            df (pd.DataFrame): Original dataframe.
            new_df (pd.DataFrame): Modified dataframe with simulation data.
            criteria_name (str): Column name for simulation criteria.
        """
        cstdev = SGS.local_standard_deviation(realizations,nx,ny)
        SGS.loc_grid(
            cstdev, df['LongDegree'].min(), df['LongDegree'].max(),
            df['LatDegree'].min(), df['LatDegree'].max(), 1, 100, new_df, 'X', 'Y', criteria_name,
            'PostSIM - Cond. St.Dev. Model', 'X(Longitude)', 'Y(Latitude)', 'Cond. StDev.', cmap
        )
