[![DOI](https://zenodo.org/badge/710838372.svg)](https://zenodo.org/doi/10.5281/zenodo.10053554)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# StraitFlux
Tools for easy and precise computations of **oceanic transports of volume, heat, salinity, sea ice and arbitrary tracers**, as well as **vertical cross-sections of currents, temperature and salinity**. StraitFlux additionally provides routines for calculating **depth- and density-space overturning circulation (MOC), water-mass transports, gyre and overturning heat/salt decomposition, and optional barotropic transport adjustment**.

StraitFlux works on regular and curvilinear ocean-model grids, independent of the exact grid curvature, the number of poles, or the employed Arakawa staggering. Calculations are performed on the **native model grid without interpolation**, enabling accurate transport estimates even for highly distorted grids. More information may be found in Winkelbauer et al. (2024).

# Main capabilities
- Accurate transport calculations on native model grids
- Volume, heat, salinity, sea-ice and arbitrary tracer transports
- Exact line-integration and vector-projection methods
- Vertical cross-sections of velocity, temperature, salinity and tracers
- Depth-space and density-space overturning circulation (MOC)
- Gyre and overturning decomposition of heat and salt transports
- Water-mass transport decomposition using user-defined criteria
- Optional barotropic transport adjustment
- Support for regular, curvilinear, tripolar and displaced-pole grids
- Automatic detection of Arakawa A-, B- and C-grid staggering
- File-based and xarray Dataset-based workflows

# Key Publications
Winkelbauer, S., Mayer, M., and Haimberger, L.: StraitFlux – precise computations of water strait fluxes on various modeling grids, Geosci. Model Dev., 17, 4603–4620, https://doi.org/10.5194/gmd-17-4603-2024, 2024.

# Requirements and Installation
StraitFlux is written in Python and requires python >=3.10. It was tested on Python 3.10.13 and 3.11.6.

The following packages have to be installed:<br>
`xarray`<br>
`netcdf4`<br>
`xesmf`<br>
`xmip`<br>
`tqdm`<br>
To use all functions and improve the performance also add:<br>
`matplotlib`<br>
`dask`<br>

You may download StraitFlux via pypi by running `pip install StraitFlux`. <br>
**Note that ESMpy is not available via pypi and has to be installed prior to StraitFlux using conda!**, you may run for instance:<br>
`conda create -n ENVNAME python=3.11.6 xesmf`<br>
`conda activate ENVNAME`<br>
`pip install StraitFlux`<br>

Alternatively you may download all needed packages by e.g. running: `conda create -n StraitFlux python=3.11.6 xarray netcdf4 xesmf xmip tqdm matplotlib dask` <br>

# Usage and testing
`Examples1.ipynb` and `Examples2.ipynb` contain some easy examples to get started with the calculations (written fro a UNIX-environment). <br>
Data files used in the notebook may be downloaded using `Download.ipynb` via ESGF (https://esgf-node.llnl.gov/search/cmip6/) for CMIP6 and the Mercator Ocean OpenDAP system for reanalyses. You'll need about 3.7GB for the used CanESM5 files and about 16GB for the used reanalyses files.


# Attribution
If you use StraitFlux in scientific work, please cite:

Winkelbauer, S., Mayer, M., and Haimberger, L. (2024):
StraitFlux – precise computations of water strait fluxes on various modeling grids.
Geosci. Model Dev., 17, 4603–4620.
https://doi.org/10.5194/gmd-17-4603-2024

# License
StraitFlux is a free software and can be redistributed and/or modified under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.
