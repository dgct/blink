import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="BLINK",
    version="3.1.4",
    author="Daniel Treen",
    author_email="d@tree.casa",
    description=(
        "Blur and Link (BLINK) is a Python library that acts as an abstraction of sparse matrices to enable fast and efficient cosine scoring across noisy dimensions."
    ),
    license="BSD",
    keywords="blur link vector",
    url="",
    packages=["blink"],
    install_requires=["numpy", "scipy"],
    long_description=read("README.md"),
    classifiers=[],
)
