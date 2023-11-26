"Setup package."
from setuptools import find_packages, setup

setup(
    name="gnnepcsaft",
    version="1.0.0",
    description="GNNePCSAFT Project.",
    url="https://github.com/wildsonbbl/gnnepcsaft.git",
    author="Wildson Lima",
    author_email="wil_bbl@hotmail.com",
    license="GNU",
    packages=find_packages(),
    requires=["torch", "torch_geometric"],
    zip_safe=False,
)
