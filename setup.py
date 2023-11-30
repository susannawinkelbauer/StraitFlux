from setuptools import setup

setup(
    name = 'StraitFlux',
    packages = ['StraitFlux'],
    version = '1.0.1',
    install_requires=['xarray', 'netcdf4', 'xesmf', 'xmip', 'tqdm', 'matplotlib', 'dask'], #external packages as dependencies
    python_requires=">=3.10",
    description = 'StraitFlux',
    long_description = 'StraitFlux - calculates oceanic transports and crosssections on different model grids.',
    author = 'Susanna Winkelbauer',
    author_email = 'susanna.winkelbauer@univie.ac.at',
    url = 'https://github.com/susannawinkelbauer/StraitFlux',
    download_url = 'https://github.com/susannawinkelbauer/StraitFlux/archive/refs/tags/v1.0.1.tar.gz',
    keywords = ['python', 'oceanic transports','curvilinear grids'],
    classifiers = ["Intended Audience :: Science/Research", "Programming Language :: Python :: 3",],
)
