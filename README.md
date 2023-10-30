[![DOI](https://zenodo.org/badge/710838372.svg)](https://zenodo.org/doi/10.5281/zenodo.10053554)

# StraitFlux

Tools for easy and precise computations of **oceanic transports of volume, heat, salinity and ice**, as well as **crosssections** of the vertical plane of **currents, temperature and salinity**. StraitFlux works on various curvilinear modelling grids (+ regular grids), independant of the exact curvature, the number of poles and the used Arakawa partition. More information may be found in Winkelbauer et al. (2023).

# Key Publications
Winkelbauer, S., Mayer, and Haimberger, L.: StraitFlux - Precise computations of Water Strait fluxes on various Modelling Grids. In preperation.

# Requiremnets and Installation
StraitFlux is written in Python and requires at least version 3.8. It is currently run and tested in 3.11.6.

The following package have to be installed:<br>
`xarray=2022.12.0`<br>
`netcdf4`<br>
`xesmf=0.7.0`<br>
`xmip=0.6.1`<br>
`tqdm`<br>
To use all functions and improve the performance also add:<br>
`matplotlib=3.6.2`<br>
`dask=2022.12.1`<br>

you may e.g. run: `conda create -n StraitFlux python=3.11.6 xarray=2022.12.0 netcdf4 xesmf=0.7.0 xmip=0.6.1 tqdm` <br>

Activate your environment and start calculations.

# Usage and testing
`Examples.ipynb` contains some easy examples to get started with the calculations. Data files used in the notebook may be downloaded via ESGF (https://esgf-node.llnl.gov/search/cmip6/).


# Attribution
If you use the code please cite Winkelbauer et al. (2023, see above).

# License
StraitFlux is a free software and can be redistributed and/or modified under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.
