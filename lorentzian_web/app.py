"""
Lorentzian Fitting Web Application

A web interface for uploading astronomical data and performing automated
Lorentzian profile fitting with model selection.
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import uuid

# Import our fitting package
import sys
sys.path.append('/Users/rahulgupta/Developer/astro')
from lorentzian_fitting.pipeline import AutomatedPipeline, PipelineSettings

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv', 'dat', 'tsv'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class DataValidator:
    """Class for validating uploaded data."""
    
    @staticmethod
    def validate_data(df):
        """
        Validate uploaded data for Lorentzian fitting.
        
        Returns:
        --------
        dict: Validation results with status, errors, warnings, and info
        """
        errors = []
        warnings = []
        info = []
        
        # Check basic structure
        if df.empty:
            errors.append("File is empty")
            return {'status': 'error', 'errors': errors, 'warnings': warnings, 'info': info}
        
        if len(df.columns) < 2:
            errors.append("Need at least 2 columns (x and y values)")
            return {'status': 'error', 'errors': errors, 'warnings': warnings, 'info': info}
        
        # Check data types and convert if needed
        numeric_cols = []
        for i, col in enumerate(df.columns[:3]):  # Check first 3 columns
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                if i < 2:  # First two columns must be numeric
                    errors.append(f"Column '{col}' contains non-numeric data")
        
        if len(numeric_cols) < 2:
            errors.append("Need at least 2 numeric columns")
            return {'status': 'error', 'errors': errors, 'warnings': warnings, 'info': info}
        
        # Extract x, y, and optional yerr
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        yerr_col = numeric_cols[2] if len(numeric_cols) >= 3 else None
        
        x = df[x_col].dropna()
        y = df[y_col].dropna()
        
        # Check for sufficient data
        if len(x) != len(y):
            errors.append("X and Y columns have different lengths after removing NaN values")
        
        if len(x) < 10:
            errors.append("Need at least 10 data points for reliable fitting")
        elif len(x) < 20:
            warnings.append("Less than 20 data points - results may be unreliable")
        
        # Check for NaN values
        n_nan_x = df[x_col].isna().sum()
        n_nan_y = df[y_col].isna().sum()
        
        if n_nan_x > 0 or n_nan_y > 0:
            warnings.append(f"Removed {max(n_nan_x, n_nan_y)} rows with NaN values")
        
        # Check data ranges
        if len(x) > 0 and len(y) > 0:
            x_range = np.ptp(x)
            y_range = np.ptp(y)
            
            if x_range == 0:
                errors.append("X values are all identical")
            if y_range == 0:
                warnings.append("Y values are all identical - may indicate constant baseline")
            
            # Check for reasonable dynamic range
            if y_range < 1e-10:
                warnings.append("Very small Y range detected - may cause numerical issues")
            
            # Check for potential outliers
            y_median = np.median(y)
            y_mad = np.median(np.abs(y - y_median))
            if y_mad > 0:
                outliers = np.abs(y - y_median) > 5 * y_mad
                n_outliers = np.sum(outliers)
                if n_outliers > 0:
                    warnings.append(f"Detected {n_outliers} potential outliers (>5σ from median)")
        
        # Check error column if present
        if yerr_col is not None:
            yerr = df[yerr_col].dropna()
            if len(yerr) != len(y):
                warnings.append("Error column has different length - will be ignored")
                yerr_col = None
            elif np.any(yerr <= 0):
                warnings.append("Error column contains non-positive values")
            elif np.any(yerr > y_range):
                warnings.append("Some error values are larger than the data range")
        
        # Generate info
        info.append(f"Data shape: {len(x)} points")
        info.append(f"X range: {np.min(x):.3g} to {np.max(x):.3g}")
        info.append(f"Y range: {np.min(y):.3g} to {np.max(y):.3g}")
        if yerr_col is not None:
            info.append(f"Error column detected: {yerr_col}")
        
        # Estimate signal-to-noise ratio
        if yerr_col is not None:
            snr = np.ptp(y) / np.median(yerr)
            info.append(f"Estimated S/N ratio: {snr:.1f}")
            if snr < 5:
                warnings.append("Low signal-to-noise ratio detected")
        
        # Determine overall status
        if errors:
            status = 'error'
        elif warnings:
            status = 'warning'
        else:
            status = 'success'
        
        return {
            'status': status,
            'errors': errors,
            'warnings': warnings,
            'info': info,
            'columns': {
                'x': x_col,
                'y': y_col,
                'yerr': yerr_col
            },
            'data_preview': {
                'n_points': len(x),
                'x_range': [float(np.min(x)), float(np.max(x))],
                'y_range': [float(np.min(y)), float(np.max(y))]
            }
        }

@app.route('/')
def index():
    """Main page with data upload interface."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and validation."""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error', 
            'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        })
    
    try:
        # Generate unique session ID for this upload
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Read file based on extension
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(file)
        elif file_ext in ['txt', 'dat', 'tsv']:
            # Try different delimiters
            try:
                df = pd.read_csv(file, delimiter='\t')
            except:
                try:
                    file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file, delimiter=' ', skipinitialspace=True)
                except:
                    file.seek(0)
                    df = pd.read_csv(file, delimiter=None, engine='python')
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'})
        
        # Validate data
        validator = DataValidator()
        validation_result = validator.validate_data(df)
        
        if validation_result['status'] != 'error':
            # Store data in session (for small datasets) or save to file
            if len(df) < 10000:  # Store small datasets in session
                session['data'] = df.to_json()
                session['validation'] = validation_result
            else:
                # Save large datasets to file
                data_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}.csv')
                df.to_csv(data_file, index=False)
                session['data_file'] = data_file
                session['validation'] = validation_result
            
            # Generate preview plot
            plot_url = generate_preview_plot(df, validation_result['columns'])
            validation_result['plot_url'] = plot_url
        
        return jsonify(validation_result)
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': f'Error reading file: {str(e)}'
        })

def generate_preview_plot(df, columns):
    """Generate a preview plot of the uploaded data."""
    try:
        plt.figure(figsize=(10, 6))
        
        x = df[columns['x']].dropna()
        y = df[columns['y']].dropna()
        
        # Ensure same length
        min_len = min(len(x), len(y))
        x = x.iloc[:min_len]
        y = y.iloc[:min_len]
        
        if columns['yerr'] is not None:
            yerr = df[columns['yerr']].dropna().iloc[:min_len]
            if len(yerr) == len(x) and np.all(yerr > 0):
                plt.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.7, markersize=4)
            else:
                plt.plot(x, y, 'o', alpha=0.7, markersize=4)
        else:
            plt.plot(x, y, 'o', alpha=0.7, markersize=4)
        
        plt.xlabel(f'{columns["x"]} (Column 1)')
        plt.ylabel(f'{columns["y"]} (Column 2)')
        plt.title('Data Preview')
        plt.grid(True, alpha=0.3)
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plot_url = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_url}"
        
    except Exception as e:
        plt.close()
        return None

@app.route('/clear')
def clear_session():
    """Clear current session data."""
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/data_info')
def data_info():
    """Get information about currently loaded data."""
    if 'validation' not in session:
        return jsonify({'status': 'error', 'message': 'No data loaded'})
    
    return jsonify(session['validation'])

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
