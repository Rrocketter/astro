"""
Quick runner script for the Lorentzian Fitting Web Application
"""

from app import app

if __name__ == '__main__':
    print("Starting Lorentzian Fitting Web Application...")
    print("Open your browser and go to: http://localhost:5008")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5008)
