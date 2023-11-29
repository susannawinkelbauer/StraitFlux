[![DOI](https://zenodo.org/badge/710838372.svg)](https://zenodo.org/doi/10.5281/zenodo.10053554)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# StraitFlux

Tools for easy and precise computations of **oceanic transports of volume, heat, salinity and ice**, as well as **crosssections** of the vertical plane of **currents, temperature and salinity**. StraitFlux works on various curvilinear modelling grids (+ regular grids), independant of the exact curvature, the number of poles and the used Arakawa partition. More information may be found in Winkelbauer et al. (2023).

# Key Publications
Winkelbauer, S., Mayer, and Haimberger, L.: StraitFlux - Precise computations of Water Strait fluxes on various Modelling Grids. In preperation.

# Requiremnets and Installation
StraitFlux is written in Python and requires python >=3.10. It was tested on Python 3.10.13 and 3.11.6.

The following package have to be installed:<br>
`xarray`<br>
`netcdf4`<br>
`xesmf`<br>
`xmip`<br>
`tqdm`<br>
To use all functions and improve the performance also add:<br>
`matplotlib`<br>
`dask`<br>

you may e.g. run: `conda create -n StraitFlux python=3.11.6 xarray netcdf4 xesmf xmip tqdm matplotlib dask` <br>

Activate your environment and start calculations.

# Usage and testing
`Examples.ipynb` contains some easy examples to get started with the calculations. Data files used in the notebook may be downloaded via ESGF (https://esgf-node.llnl.gov/search/cmip6/).


# Attribution
If you use the code please cite Winkelbauer et al. (2023, see above).

# License
StraitFlux is a free software and can be redistributed and/or modified under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.
