"""
Setup script for Fashion Recognition System
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-recognition-system",
    version="1.0.0",
    author="Fashion Recognition Team",
    author_email="contact@fashionrecognition.com",
    description="A comprehensive fashion recognition system for real-time clothing detection and brand identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clothingvision",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-recognition=fashion_recognition_system:main",
            "fashion-ui=fashion_ui:main",
            "fashion-train=model_training:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords="fashion recognition, computer vision, clothing detection, brand identification, deep learning, pytorch, opencv",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/clothingvision/issues",
        "Source": "https://github.com/yourusername/clothingvision",
        "Documentation": "https://fashion-recognition-system.readthedocs.io/",
    },
)
