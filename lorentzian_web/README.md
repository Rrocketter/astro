# Lorentzian Fitting Tool

A web-based tool for automated Lorentzian profile fitting of astronomical data.

## Features

- **Data Upload & Validation**: Upload CSV, TXT, DAT, or TSV files with automatic validation
- **Interactive Visualization**: Explore your data with zoom, pan, and peak detection
- **Automated Fitting**: Configurable pipeline with model selection using AIC/BIC criteria
- **Professional Results**: Detailed parameter tables, plots, and multiple export formats

## Usage

1. Upload your data file (wavelength, flux, optional error columns)
2. Visualize and explore your data interactively
3. Configure fitting parameters and model selection criteria
4. View detailed results and export in JSON, CSV, or TXT format

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`

## Deployment

This app is configured for deployment on Render.com using the included `render.yaml` configuration.

## Data Format

Expected format:
```
wavelength,flux,error
6562.0,1.02,0.05
6562.2,1.05,0.05
...
```

## Requirements

- Python 3.8+
- Flask 2.3+
- NumPy, SciPy, Pandas
- Matplotlib for plotting
- Custom Lorentzian fitting package
