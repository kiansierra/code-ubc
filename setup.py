# As pere https://github.com/qubvel/segmentation_models.pytorch/blob/master/setup.py
import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "ubc"
DESCRIPTION = "Kaggle-UBC-OCEAN competition. PyTorch. https://www.kaggle.com/competitions/UBC-OCEAN/overview"
URL = "https://github.com/kiansierra/code-ubc"
EMAIL = "kiansierra90@gmail.com"
AUTHOR = "Kian Sierra McGettigan"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = None
here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    REQUIRED = []

# What packages are optional?
EXTRAS = {"test": ["pytest", "mock", "flake8==4.0.1", "flake8-docstrings==1.6.0"]}

about = {}
if not VERSION:
    with open(os.path.join(here, "src", NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*/*.yaml"]},
    install_requires=[
        "requests",
        'importlib-metadata; python_version == "3.8"',
    ],
    extras_require=EXTRAS,
    include_package_data=True,
)
