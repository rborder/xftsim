from setuptools import setup, find_packages

setup(
    name='xftsim',
    
    version='0.1.9',
    
    author="Richard Border",
    
    author_email="border.richard@gmail.com",
    
    packages=find_packages(include=[
        'xftsim', 'xftsim*',
    ]),
    
    install_requires = [
    "funcy",
    "networkx",
    "nptyping",
    "numba==0.56.4",
    "numpy",
    "pandas",
    "pandas_plink",
    "scipy",
    "sgkit",
    "xarray",
    ],

    include_package_data=True,
)
