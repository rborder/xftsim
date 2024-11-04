from setuptools import setup, find_packages

setup(
    name='xftsim',
    
    version='0.2.0',
    
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

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
