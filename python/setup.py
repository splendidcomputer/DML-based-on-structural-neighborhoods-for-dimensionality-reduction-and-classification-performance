"""
Setup script for the Distance Metric Learning Python implementation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()
                    and not line.startswith("#")]

setup(
    name="distance-metric-learning",
    version="1.0.0",
    author="Python Implementation",
    description="Python implementation of Distance Metric Learning based on structural neighborhoods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/splendidcomputer/DML-based-on-structural-neighborhoods-for-dimensionality-reduction-and-classification-performance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dml-experiment=main:main",
        ],
    },
)
