from setuptools import setup, find_packages

setup(
    name="merit",
    version="0.1.0",
    description="Multi-dimensional Evaluation of Reasoning in Transformers",
    author="MERIT Team",
    packages=find_packages(include=["merit", "merit.*"]),
    install_requires=[
        "huggingface_hub>=0.19.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "evaluate>=0.4.0",
        "sentence-transformers>=2.2.2",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
    ]
)
