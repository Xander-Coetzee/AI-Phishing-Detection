import subprocess
import os
import sys

def main():
    # Set environment variables
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_SERVER_ADDRESS"] = "0.0.0.0"
    
    # Command to run Streamlit
cmd = [
    sys.executable,  # Current Python interpreter
    "-m", "streamlit", "run",
    "--server.port", "8765",
   # Only disable protections in development environments
   # "--server.enableCORS", "false",
   # "--server.enableXsrfProtection", "false",
    "--logger.level", "debug",
    "app.py"
]
    
    print("Starting Streamlit app...")
    print("Access the app at: http://localhost:8765")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
