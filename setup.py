from setuptools import setup, find_packages

setup(
    name="lorentzian_fitting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "lmfit>=1.2.0",
        "flask>=3.0.0"
    ],
    python_requires=">=3.9",
    author="Your Name",
    description="Lorentzian function fitting for astronomical data",
)
