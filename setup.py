from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='molmamba',
    version="0.0.1",
    packages=find_packages(),
    # metadata
    author='Yiming Zhang',
    description="",
    license="Apache License 2.0",
    include_package_data=True,
    install_requires=required,
)
