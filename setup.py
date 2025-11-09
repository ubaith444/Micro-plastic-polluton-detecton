"""Setup script"""

from setuptools import setup, find_packages

setup(
    name="underwater-plastic-detection",
    version="1.0.0",
    author="Underwater Detection Team",
    description="CNN-based underwater plastic detection using UPD dataset",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'ultralytics>=8.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.62.0'
    ],
)
