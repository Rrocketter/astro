from setuptools import setup, find_packages

setup(
    name="lorentzian_fitting",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "lmfit>=1.2.0",
        "flask>=3.0.0",
        "gunicorn>=21.2.0",
    ],
    python_requires=">=3.9,<3.13",
    author="Rahul Gupta",
    description="Lorentzian function fitting for astronomical data",
    extras_require={
        "data": ["pandas>=2.1.0,<2.2.0"],
        "dev": ["pytest", "black", "flake8"]
    }
)
