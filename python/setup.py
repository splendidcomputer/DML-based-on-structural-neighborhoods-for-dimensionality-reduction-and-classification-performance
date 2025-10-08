""""""

Setup script for Distance Metric Learning Python implementation.Setup script for the Distance Metric Learning Python implementation.

""""""



from setuptools import setup, find_packagesfrom setuptools import setup, find_packages



with open("README.md", "r", encoding="utf-8") as fh:with open("README.md", "r", encoding="utf-8") as fh:

    long_description = fh.read()    long_description = fh.read()



with open("requirements.txt", "r", encoding="utf-8") as fh:with open("requirements.txt", "r", encoding="utf-8") as fh:

    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]    requirements = [line.strip() for line in fh if line.strip()

                    and not line.startswith("#")]

setup(

    name="distance-metric-learning",setup(

    version="1.0.0",    name="distance-metric-learning",

    author="Python Implementation Team",    version="1.0.0",

    author_email="",    author="Python Implementation",

    description="Distance Metric Learning based on Structural Neighborhoods",    description="Python implementation of Distance Metric Learning based on structural neighborhoods",

    long_description=long_description,    long_description=long_description,

    long_description_content_type="text/markdown",    long_description_content_type="text/markdown",

    url="https://github.com/yourusername/DML-structural-neighborhoods",    url="https://github.com/splendidcomputer/DML-based-on-structural-neighborhoods-for-dimensionality-reduction-and-classification-performance",

    packages=find_packages(),    packages=find_packages(),

    classifiers=[    classifiers=[

        "Development Status :: 4 - Beta",        "Development Status :: 4 - Beta",

        "Intended Audience :: Science/Research",        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: MIT License",        "License :: OSI Approved :: MIT License",

        "Operating System :: OS Independent",        "Operating System :: OS Independent",

        "Programming Language :: Python :: 3",        "Programming Language :: Python :: 3",

        "Programming Language :: Python :: 3.7",        "Programming Language :: Python :: 3.7",

        "Programming Language :: Python :: 3.8",        "Programming Language :: Python :: 3.8",

        "Programming Language :: Python :: 3.9",        "Programming Language :: Python :: 3.9",

        "Programming Language :: Python :: 3.10",        "Programming Language :: Python :: 3.10",

        "Programming Language :: Python :: 3.11",        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        "Topic :: Scientific/Engineering :: Artificial Intelligence",        "Topic :: Scientific/Engineering :: Information Analysis",

        "Topic :: Scientific/Engineering :: Information Analysis",    ],

    ],    python_requires=">=3.7",

    python_requires=">=3.7",    install_requires=requirements,

    install_requires=requirements,    extras_require={

    extras_require={        "dev": [

        "dev": [            "pytest>=6.0",

            "pytest>=6.0",            "black>=21.0",

            "pytest-cov>=2.0",            "flake8>=3.8",

            "black>=21.0",            "jupyter>=1.0",

            "flake8>=3.8",        ],

        ],    },

        "docs": [    entry_points={

            "sphinx>=4.0",        "console_scripts": [

            "sphinx-rtd-theme>=1.0",            "dml-experiment=main:main",

        ],        ],

    },    },

    entry_points={)

        "console_scripts": [
            "dml-run=main:main",
        ],
    },
)