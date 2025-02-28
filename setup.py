from setuptools import setup, find_packages

setup(
    name="parkinsons_detection",
    version="1.0.0",
    description="Parkinson's Disease Detection using Spiral and Wave Drawing Tests",
    author="Advanced ML Team",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "opencv-python>=4.5.0",
        "joblib>=1.0.0",
        "imbalanced-learn>=0.8.0",
        "tqdm>=4.60.0",
        "gradio>=3.0.0",
        "pillow>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pddetect=main:init_command",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
