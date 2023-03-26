from setuptools import setup, find_packages

setup(
    name='xftsim',
    
    version='0.0.0',
    
    author="Richard Border",
    
    author_email="border.richard@gmail.com",
    
    packages=find_packages(include=[
        'xftsim', 'xftsim*',
    ]),
    
    install_requires = [
    "networkx",
    "nptyping",
    "numba",
    "numpy",
    "pandas",
    "scipy",
    "sgkit",
    "xarray",
    ],

)
