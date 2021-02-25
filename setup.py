import io
import os
import setuptools


# ==============================================================================
# Utilities
# ==============================================================================
def read(path, encoding="utf-8"):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()


# ==============================================================================
# Variables
# ==============================================================================
DESCRIPTION = "Machine Learning Algorithms from Scratch in Python"
LONG_DESCRIPTION = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
NAME = "mlalgs"
AUTHOR = "Amirhessam Tahmassebi"
AUTHOR_EMAIL = "amir.benny@gmail.com"
URL = "http://www.amirhessam.com"
DOWNLOAD_URL = "https://github.com/amirhessam88/ml-algs/"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.6"
PACKAGES = setuptools.find_packages()
VERSION = "0.0.2"

# ==============================================================================
# Setup
# ==============================================================================

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    packages=PACKAGES,
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
)
