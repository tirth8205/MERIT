from setuptools import setup, find_packages
import os


def read_long_description():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="merit",
    version="3.0.0",
    description="Multi-dimensional Evaluation of Reasoning in Transformers",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Tirth Patel",
    python_requires=">=3.9",
    packages=find_packages(include=["merit", "merit.*"]),
    install_requires=[
        "huggingface_hub>=0.19.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "spacy>=3.6.0",
        "nltk>=3.8.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "wikipedia>=1.4.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "judge": [
            "anthropic>=0.40.0",
        ],
        "baselines": [
            "bert-score>=0.3.13",
        ],
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "all": [
            "anthropic>=0.40.0",
            "bert-score>=0.3.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "merit=merit.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="nlp evaluation reasoning transformers llm metrics",
    include_package_data=True,
    zip_safe=False,
)
