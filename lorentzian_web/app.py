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
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session lifetime

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
        session.permanent = True  # Make session persistent
        
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
            # Store data in file
            data_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_data.csv')
            df.to_csv(data_file, index=False)
            
            # Store only minimal metadata in session (no large objects)
            session['session_id'] = session_id
            session['data_file'] = data_file
            session['filename'] = filename
            session['data_shape'] = df.shape
            session['status'] = validation_result['status']
            session['columns'] = validation_result['columns']
            session['data_preview'] = validation_result.get('data_preview', {})
            
            # Store validation messages separately (they might be large)
            validation_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_validation.json')
            import json
            with open(validation_file, 'w') as f:
                json.dump(validation_result, f)
            session['validation_file'] = validation_file
            
            print(f"Stored data in file: {data_file}")
            print(f"Stored validation in file: {validation_file}")
            print(f"Data shape: {df.shape}")
            print(f"Session ID: {session_id}")
            print(f"Session size: {len(str(dict(session)))} characters")
            
            # Generate preview plot
            plot_url = generate_preview_plot(df, validation_result['columns'])
            validation_result['plot_url'] = plot_url
        
        return jsonify(validation_result)
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
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

@app.route('/api/data_info')
def data_info():
    """Get information about currently loaded data."""
    print("data_info called, session keys:", list(session.keys()))
    print(f"Session size: {len(str(dict(session)))} characters")
    
    if 'session_id' not in session:
        return jsonify({'status': 'error', 'message': 'No session ID found'})
    
    if 'data_file' not in session:
        return jsonify({'status': 'error', 'message': 'No data file in session'})
    
    # Check if data file still exists
    data_file = session['data_file']
    if not os.path.exists(data_file):
        return jsonify({'status': 'error', 'message': 'Data file no longer exists'})
    
    # Load validation data from file
    validation_file = session.get('validation_file')
    if validation_file and os.path.exists(validation_file):
        import json
        with open(validation_file, 'r') as f:
            validation = json.load(f)
    else:
        # Fallback to minimal validation data
        validation = {
            'status': session.get('status', 'unknown'),
            'columns': session.get('columns', {}),
            'data_preview': session.get('data_preview', {})
        }
    
    return jsonify({
        'status': 'success',
        'validation': validation,
        'data_preview': session.get('data_preview', {}),
        'columns': session.get('columns', {}),
        'filename': session.get('filename', 'Unknown'),
        'data_shape': session.get('data_shape', [0, 0])
    })

@app.route('/api/get_data')
def get_data():
    """Get the current data for visualization."""
    print("get_data called, session keys:", list(session.keys()))
    
    if 'session_id' not in session:
        return jsonify({'status': 'error', 'message': 'No session ID found'})
    
    if 'data_file' not in session:
        return jsonify({'status': 'error', 'message': 'No data file found in session'})
    
    try:
        # Load data from file
        data_file = session['data_file']
        
        if not os.path.exists(data_file):
            return jsonify({'status': 'error', 'message': 'Data file not found on server'})
        
        df = pd.read_csv(data_file)
        
        # Load validation from file
        validation_file = session.get('validation_file')
        if validation_file and os.path.exists(validation_file):
            import json
            with open(validation_file, 'r') as f:
                validation = json.load(f)
        else:
            # Create minimal validation object
            validation = {
                'status': session.get('status', 'success'),
                'columns': session.get('columns', {}),
                'data_preview': session.get('data_preview', {})
            }
        
        columns = validation['columns']
        
        print(f"Loaded data from file: {data_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {columns}")
        
        # Extract data arrays
        x = df[columns['x']].dropna().values
        y = df[columns['y']].dropna().values
        
        # Ensure same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Get error column if available
        yerr = None
        if columns['yerr'] is not None and columns['yerr'] in df.columns:
            yerr_data = df[columns['yerr']].dropna().values
            if len(yerr_data) >= min_len:
                yerr = yerr_data[:min_len].tolist()
        
        # Estimate potential peaks for visualization
        peaks_info = estimate_peaks_for_viz(x, y)
        
        return jsonify({
            'status': 'success',
            'data': {
                'x': x.tolist(),
                'y': y.tolist(),
                'yerr': yerr,
                'columns': columns,
                'n_points': len(x)
            },
            'peaks': peaks_info,
            'validation': validation
        })
        
    except Exception as e:
        print(f"get_data error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Error loading data: {str(e)}'})

@app.route('/clear')
def clear_session():
    """Clear current session data and files."""
    # Clean up data files if they exist
    for file_key in ['data_file', 'validation_file']:
        if file_key in session:
            file_path = session[file_key]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing file: {e}")
    
    session.clear()
    return redirect(url_for('index'))

@app.route('/visualize')
def visualize():
    """Interactive visualization page."""
    # Debug: Check what's in session
    print("Session contents:", dict(session))
    
    # Check for session_id instead of validation
    if 'session_id' not in session:
        print("No session_id in session, redirecting to index")
        return redirect(url_for('index'))
    
    # Check if data file exists
    if 'data_file' not in session:
        print("No data_file in session, redirecting to index")
        return redirect(url_for('index'))
    
    data_file = session['data_file']
    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist, redirecting to index")
        return redirect(url_for('index'))
    
    print("Data confirmed, rendering visualize template")
    return render_template('visualize.html')

@app.route('/api/analyze_region', methods=['POST'])
def analyze_region():
    """Analyze a selected region of the data."""
    try:
        data = request.get_json()
        x_min = data.get('x_min')
        x_max = data.get('x_max')
        
        if not all([x_min is not None, x_max is not None]):
            return jsonify({'status': 'error', 'message': 'Invalid region selection'})
        
        # Get current data
        if 'data' in session:
            df = pd.read_json(session['data'])
        elif 'data_file' in session:
            df = pd.read_csv(session['data_file'])
        else:
            return jsonify({'status': 'error', 'message': 'No data found'})
        
        validation = session['validation']
        columns = validation['columns']
        
        # Filter data to selected region
        x_full = df[columns['x']].dropna().values
        y_full = df[columns['y']].dropna().values
        
        mask = (x_full >= x_min) & (x_full <= x_max)
        x_region = x_full[mask]
        y_region = y_full[mask]
        
        if len(x_region) < 5:
            return jsonify({'status': 'error', 'message': 'Selected region has too few points'})
        
        # Analyze the region
        region_analysis = analyze_data_region(x_region, y_region)
        
        return jsonify({
            'status': 'success',
            'analysis': region_analysis,
            'region': {
                'x_min': x_min,
                'x_max': x_max,
                'n_points': len(x_region)
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Analysis error: {str(e)}'})

def estimate_peaks_for_viz(x, y):
    """Estimate potential peaks for visualization hints."""
    try:
        from scipy.signal import find_peaks
        
        # Remove baseline trend
        y_detrended = y - np.median(y)
        
        # Find peaks with reasonable prominence
        prominence = np.std(y_detrended) * 0.3
        peaks, properties = find_peaks(y_detrended, prominence=prominence, width=1)
        
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append({
                'x': float(x[peak_idx]),
                'y': float(y[peak_idx]),
                'prominence': float(properties['prominences'][i]),
                'width': float(properties['widths'][i])
            })
        
        return {
            'detected_peaks': peak_info,
            'n_peaks': len(peaks),
            'baseline_estimate': float(np.median(y))
        }
        
    except Exception:
        return {
            'detected_peaks': [],
            'n_peaks': 0,
            'baseline_estimate': float(np.median(y)) if len(y) > 0 else 0.0
        }

def analyze_data_region(x, y):
    """Analyze a specific region of data."""
    analysis = {
        'statistics': {
            'x_range': [float(np.min(x)), float(np.max(x))],
            'y_range': [float(np.min(y)), float(np.max(y))],
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)),
            'n_points': len(x)
        }
    }
    
    # Estimate signal characteristics
    y_baseline = np.median(y)
    y_peak = np.max(y)
    signal_amplitude = y_peak - y_baseline
    
    analysis['signal'] = {
        'baseline': float(y_baseline),
        'peak': float(y_peak),
        'amplitude': float(signal_amplitude),
        'snr_estimate': float(signal_amplitude / np.std(y)) if np.std(y) > 0 else 0
    }
    
    # Suggest number of components
    if signal_amplitude < 2 * np.std(y):
        suggested_components = 0  # Noise only
    else:
        # Simple peak counting
        from scipy.signal import find_peaks
        try:
            y_detrended = y - y_baseline
            peaks, _ = find_peaks(y_detrended, prominence=signal_amplitude * 0.1)
            suggested_components = min(len(peaks), 3)  # Cap at 3 components
        except:
            suggested_components = 1
    
    analysis['suggestions'] = {
        'n_components': suggested_components,
        'fit_quality_expectation': 'good' if analysis['signal']['snr_estimate'] > 5 else 'challenging'
    }
    
    return analysis

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB.'
    }), 413

# Add cleanup function for old files
def cleanup_old_files():
    """Clean up old data files (older than 1 hour)."""
    import time
    
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return
    
    current_time = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('_data.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file_age = current_time - os.path.getctime(file_path)
            
            # Remove files older than 1 hour
            if file_age > 3600:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Error cleaning up {filename}: {e}")

# Call cleanup on startup
cleanup_old_files()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
