from setuptools import setup, find_packages

setup(
    name="eeglearn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'mne',
        'pyprep',
        'matplotlib'
    ],
)