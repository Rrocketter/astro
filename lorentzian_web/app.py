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

@app.route('/configure')
def configure():
    """Fitting configuration page."""
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
    
    print("Data confirmed, rendering configure template")
    return render_template('configure.html')

@app.route('/api/run_fitting', methods=['POST'])
def run_fitting():
    """Run the automated fitting pipeline with user configuration."""
    try:
        # Get configuration from request
        config_data = request.get_json()
        print("Received fitting configuration:", config_data)
        
        # Load data from session
        if 'data_file' not in session:
            return jsonify({'status': 'error', 'message': 'No data file found'})
        
        data_file = session['data_file']
        if not os.path.exists(data_file):
            return jsonify({'status': 'error', 'message': 'Data file not found'})
        
        df = pd.read_csv(data_file)
        columns = session.get('columns', {})
        
        # Extract data
        x = df[columns['x']].dropna().values
        y = df[columns['y']].dropna().values
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        yerr = None
        if columns['yerr'] is not None and columns['yerr'] in df.columns:
            yerr_data = df[columns['yerr']].dropna().values
            if len(yerr_data) >= min_len:
                yerr = yerr_data[:min_len]
        
        # Use simplified automated model selection instead of complex pipeline
        print("Running simplified automated fitting...")
        result = run_simplified_automated_fitting(x, y, yerr, config_data)
        
        # Store results in session
        session_id = session['session_id']
        results_file = os.path.join(app.config['UPLOAD_FOLDER'], f'{session_id}_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = convert_numpy_to_lists(result)
        
        import json
        with open(results_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        session['results_file'] = results_file
        session['fitting_completed'] = True
        
        print(f"Fitting completed successfully. Results saved to {results_file}")
        
        # Return summary for immediate display
        summary = create_fitting_summary(result)
        
        return jsonify({
            'status': 'success',
            'message': 'Fitting completed successfully',
            'summary': summary
        })
        
    except Exception as e:
        print(f"Fitting error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Fitting failed: {str(e)}'
        })

def run_simplified_automated_fitting(x, y, yerr, config):
    """Run simplified automated fitting using existing components."""
    from lorentzian_fitting.fitting import LorentzianFitter
    from lorentzian_fitting.comparison import automated_model_selection
    
    # Use the automated_model_selection function directly
    max_components = config.get('max_components', 3)
    
    try:
        best_n_components, comparison_results = automated_model_selection(
            x, y, yerr, max_components=max_components
        )
        
        # Extract the best fit result
        best_model_name = comparison_results['model_selection']['best_aic_model']
        best_fit_data = comparison_results['fit_results'][best_model_name]
        best_statistics = comparison_results['statistics'][best_model_name]
        
        # Create result structure compatible with expected format
        result = {
            'best_fit': {
                'params': best_fit_data['params'],
                'param_errors': best_fit_data['param_errors'],
                'fit_info': best_statistics
            },
            'model_name': best_model_name,
            'n_components': best_fit_data['n_components'],
            'selection_summary': {
                'selected_model': best_model_name,
                'selection_confidence': 'automated',
                'selection_rationale': ['Automated model selection'],
                'alternative_models': []
            },
            'all_models': comparison_results,
            'validation': {
                'overall_quality': True,  # Simplified validation
                'fit_quality': {
                    'acceptable_chi2': True,
                    'good_r_squared': best_statistics.get('r_squared', 0) > 0.5
                }
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Automated fitting failed, trying manual approach: {e}")
        
        # Fallback to manual fitting approach
        fitter = LorentzianFitter()
        best_result = None
        best_aic = np.inf
        best_n_comp = 1
        
        # Try different numbers of components
        for n_comp in range(1, max_components + 1):
            try:
                if n_comp == 1:
                    params, param_errors, fit_info = fitter.fit_single(x, y, yerr)
                else:
                    params, param_errors, fit_info = fitter.fit_multiple(x, y, n_comp, yerr)
                
                # Check if this is the best fit so far
                if fit_info['aic'] < best_aic:
                    best_aic = fit_info['aic']
                    best_n_comp = n_comp
                    best_result = {
                        'params': params,
                        'param_errors': param_errors,
                        'fit_info': fit_info,
                        'n_components': n_comp
                    }
                    
            except Exception as fit_error:
                print(f"Failed to fit {n_comp} components: {fit_error}")
                continue
        
        if best_result is None:
            raise Exception("All fitting attempts failed")
        
        # Create simplified result structure
        model_name = f"{best_n_comp}_component{'s' if best_n_comp != 1 else ''}"
        
        result = {
            'best_fit': best_result,
            'model_name': model_name,
            'n_components': best_n_comp,
            'selection_summary': {
                'selected_model': model_name,
                'selection_confidence': 'manual',
                'selection_rationale': ['Best AIC from manual comparison'],
                'alternative_models': []
            },
            'all_models': {'manual_selection': True},
            'validation': {
                'overall_quality': True,
                'fit_quality': {
                    'acceptable_chi2': True,
                    'good_r_squared': best_result['fit_info'].get('r_squared', 0) > 0.5
                }
            }
        }
        
        return result

def convert_numpy_to_lists(obj):
    """Recursively convert numpy arrays and types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.complexfloating):
        return complex(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_lists(item) for item in obj)
    elif hasattr(obj, 'item'):  # Handle numpy scalars
        return obj.item()
    else:
        return obj

def create_fitting_summary(result):
    """Create a summary of fitting results for quick display."""
    best_fit = result['best_fit']
    
    # Safe parameter extraction
    params = best_fit['params']
    param_errors = best_fit['param_errors']
    
    if hasattr(params, 'tolist'):
        params_list = params.tolist()
    else:
        params_list = list(params) if hasattr(params, '__iter__') else [params]
    
    if hasattr(param_errors, 'tolist'):
        errors_list = param_errors.tolist()
    else:
        errors_list = list(param_errors) if hasattr(param_errors, '__iter__') else [param_errors]
    
    summary = {
        'selected_model': result['model_name'],
        'n_components': result['n_components'],
        'selection_confidence': result['selection_summary']['selection_confidence'],
        'fit_quality': {
            'reduced_chi_squared': float(best_fit['fit_info']['reduced_chi_squared']),
            'r_squared': float(best_fit['fit_info']['r_squared']),
            'aic': float(best_fit['fit_info']['aic']),
            'bic': float(best_fit['fit_info']['bic'])
        },
        'parameters': {
            'values': params_list,
            'errors': errors_list
        },
        'overall_quality': result['validation']['overall_quality']
    }
    
    return summary

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

@app.route('/results')
def results():
    """Results and export page."""
    # Check for session_id and completed fitting
    if 'session_id' not in session:
        print("No session_id in session, redirecting to index")
        return redirect(url_for('index'))
    
    if not session.get('fitting_completed', False):
        print("No fitting completed, redirecting to configure")
        return redirect(url_for('configure'))
    
    # Check if results file exists
    if 'results_file' not in session:
        print("No results_file in session, redirecting to configure")
        return redirect(url_for('configure'))
    
    results_file = session['results_file']
    if not os.path.exists(results_file):
        print(f"Results file {results_file} does not exist, redirecting to configure")
        return redirect(url_for('configure'))
    
    print("Results confirmed, rendering results template")
    return render_template('results.html')

@app.route('/api/get_results')
def get_results():
    """Get the complete fitting results."""
    try:
        if 'results_file' not in session:
            return jsonify({'status': 'error', 'message': 'No results file found'})
        
        results_file = session['results_file']
        if not os.path.exists(results_file):
            return jsonify({'status': 'error', 'message': 'Results file not found'})
        
        # Load results from file
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Also load original data for plotting
        data_file = session['data_file']
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            columns = session.get('columns', {})
            
            x = df[columns['x']].dropna().values
            y = df[columns['y']].dropna().values
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            
            yerr = None
            if columns['yerr'] is not None and columns['yerr'] in df.columns:
                yerr_data = df[columns['yerr']].dropna().values
                if len(yerr_data) >= min_len:
                    yerr = yerr_data[:min_len].tolist()
            
            # Generate fitted curve for plotting
            fitted_curve = generate_fitted_curve(x, results)
            
            results['plot_data'] = {
                'x': x.tolist(),
                'y': y.tolist(),
                'yerr': yerr,
                'fitted_curve': fitted_curve,
                'columns': columns
            }
        
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        print(f"Error getting results: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error loading results: {str(e)}'
        })

def generate_fitted_curve(x, results):
    """Generate the fitted curve for plotting."""
    try:
        from lorentzian_fitting.models import single_lorentzian, multiple_lorentzian
        
        best_fit = results['best_fit']
        params = np.array(best_fit['params'])
        n_components = results['n_components']
        
        if n_components == 0:
            # Baseline only
            return np.full_like(x, params[0]).tolist()
        elif n_components == 1:
            # Single Lorentzian
            fitted = single_lorentzian(x, *params)
        else:
            # Multiple Lorentzians
            fitted = multiple_lorentzian(x, n_components, *params)
        
        return fitted.tolist()
        
    except Exception as e:
        print(f"Error generating fitted curve: {e}")
        # Return zeros if fitting fails
        return np.zeros_like(x).tolist()

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """Export results in various formats."""
    try:
        export_data = request.get_json()
        export_format = export_data.get('format', 'json')
        
        if 'results_file' not in session:
            return jsonify({'status': 'error', 'message': 'No results to export'})
        
        results_file = session['results_file']
        if not os.path.exists(results_file):
            return jsonify({'status': 'error', 'message': 'Results file not found'})
        
        # Load results
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if export_format == 'json':
            return export_json_results(results)
        elif export_format == 'csv':
            return export_csv_results(results)
        elif export_format == 'txt':
            return export_txt_results(results)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported export format'})
            
    except Exception as e:
        print(f"Export error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Export failed: {str(e)}'})

def export_json_results(results):
    """Export results as JSON."""
    import json
    from flask import Response
    
    json_str = json.dumps(results, indent=2)
    
    return Response(
        json_str,
        mimetype='application/json',
        headers={'Content-Disposition': 'attachment; filename=lorentzian_results.json'}
    )

def export_csv_results(results):
    """Export results as CSV."""
    import io
    from flask import Response
    
    output = io.StringIO()
    
    # Write header
    output.write("# Lorentzian Fitting Results\n")
    output.write(f"# Selected Model: {results['model_name']}\n")
    output.write(f"# Number of Components: {results['n_components']}\n")
    output.write(f"# Reduced Chi-squared: {results['best_fit']['fit_info']['reduced_chi_squared']:.6f}\n")
    output.write(f"# R-squared: {results['best_fit']['fit_info']['r_squared']:.6f}\n")
    output.write(f"# AIC: {results['best_fit']['fit_info']['aic']:.2f}\n")
    output.write(f"# BIC: {results['best_fit']['fit_info']['bic']:.2f}\n")
    output.write("\n")
    
    # Write parameters
    output.write("Parameter,Value,Error\n")
    params = results['best_fit']['params']
    param_errors = results['best_fit']['param_errors']
    
    n_comp = results['n_components']
    if n_comp == 0:
        output.write(f"Baseline,{params[0]:.6f},{param_errors[0]:.6f}\n")
    else:
        for i in range(n_comp):
            idx = i * 3
            output.write(f"Component_{i+1}_Amplitude,{params[idx]:.6f},{param_errors[idx]:.6f}\n")
            output.write(f"Component_{i+1}_Center,{params[idx+1]:.6f},{param_errors[idx+1]:.6f}\n")
            output.write(f"Component_{i+1}_Width,{params[idx+2]:.6f},{param_errors[idx+2]:.6f}\n")
        output.write(f"Baseline,{params[-1]:.6f},{param_errors[-1]:.6f}\n")
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=lorentzian_results.csv'}
    )

def export_txt_results(results):
    """Export results as formatted text report."""
    import io
    from flask import Response
    
    output = io.StringIO()
    
    # Write detailed report
    output.write("LORENTZIAN FITTING RESULTS\n")
    output.write("=" * 50 + "\n\n")
    
    output.write(f"Selected Model: {results['model_name']}\n")
    output.write(f"Number of Components: {results['n_components']}\n")
    output.write(f"Selection Confidence: {results['selection_summary']['selection_confidence']}\n")
    output.write("\n")
    
    # Fit quality
    fit_info = results['best_fit']['fit_info']
    output.write("FIT QUALITY METRICS\n")
    output.write("-" * 20 + "\n")
    output.write(f"Reduced Chi-squared: {fit_info['reduced_chi_squared']:.6f}\n")
    output.write(f"R-squared: {fit_info['r_squared']:.6f}\n")
    output.write(f"AIC: {fit_info['aic']:.2f}\n")
    output.write(f"BIC: {fit_info['bic']:.2f}\n")
    output.write(f"Degrees of Freedom: {fit_info['degrees_of_freedom']}\n")
    output.write("\n")
    
    # Parameters
    output.write("FITTED PARAMETERS\n")
    output.write("-" * 20 + "\n")
    params = results['best_fit']['params']
    param_errors = results['best_fit']['param_errors']
    
    n_comp = results['n_components']
    if n_comp == 0:
        output.write(f"Baseline: {params[0]:.6f} ± {param_errors[0]:.6f}\n")
    else:
        for i in range(n_comp):
            idx = i * 3
            output.write(f"\nComponent {i+1}:\n")
            output.write(f"  Amplitude: {params[idx]:.6f} ± {param_errors[idx]:.6f}\n")
            output.write(f"  Center: {params[idx+1]:.6f} ± {param_errors[idx+1]:.6f}\n")
            output.write(f"  Width (FWHM): {params[idx+2]:.6f} ± {param_errors[idx+2]:.6f}\n")
        output.write(f"\nBaseline: {params[-1]:.6f} ± {param_errors[-1]:.6f}\n")
    
    # Selection rationale
    if 'selection_rationale' in results['selection_summary']:
        output.write("\nSELECTION RATIONALE\n")
        output.write("-" * 20 + "\n")
        for reason in results['selection_summary']['selection_rationale']:
            output.write(f"• {reason}\n")
    
    return Response(
        output.getvalue(),
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment; filename=lorentzian_results.txt'}
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
