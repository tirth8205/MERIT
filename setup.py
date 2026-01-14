from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

# Read long description from README
def read_long_description():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    name="merit",
    version="2.0.0",
    description="Multi-dimensional Evaluation of Reasoning in Transformers - Enhanced Edition",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="MERIT Team",
    author_email="merit@example.com",
    url="",  # Private research project
    packages=find_packages(include=["merit", "merit.*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.10.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "pre-commit>=3.3.2"
        ],
        "full": [
            "spacy>=3.6.0",
            "wikipedia>=1.4.0",
            "bert-score>=0.3.13",
            "krippendorff>=0.5.1"
        ]
    },
    entry_points={
        "console_scripts": [
            "merit=merit.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="nlp evaluation reasoning transformers llm metrics",
    # project_urls={
    #     "Documentation": "https://merit.readthedocs.io/",
    #     "Source": "https://github.com/yourusername/merit",
    #     "Tracker": "https://github.com/yourusername/merit/issues",
    # },
    include_package_data=True,
    zip_safe=False,
)
