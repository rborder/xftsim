from setuptools import setup, find_packages

setup(
    name='xftsim',
    version='0.0.0',
    packages=find_packages(include=[
                                    'xftsim', 'xftsim*',
                                    ])
)
