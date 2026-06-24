
from setuptools import setup, find_packages

from pathlib import Path

VERSION = '1.0.2'
DESCRIPTION = 'Image data split package for machine learning'

this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

# Setting up
setup(
    name="image-data-split",
    version=VERSION,
    author="Shouke Wei",
    author_email="shouke.wei@gmail.com",
    license='MIT License',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url = 'https://github.com/shoukewei/image_split',
    project_url = 'https://github.com/shoukewei/image_split',
    Documentation = 'https://github.com/shoukewei/image_split/blob/main/docs/example.ipynb',
    packages=find_packages(),
    keywords=['python', 'image data', 'split', 'train', 'validation','test', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)