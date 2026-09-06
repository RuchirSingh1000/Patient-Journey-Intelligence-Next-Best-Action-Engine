#!/usr/bin/env python
"""
Script to run the PharmaPulse Streamlit dashboard.
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")

    # Run streamlit
    command = [
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port=8501",
        "--server.address=localhost"
    ]

    print("Starting PharmaPulse Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")

    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()