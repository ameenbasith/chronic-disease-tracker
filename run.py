import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Run the Streamlit app
if __name__ == "__main__":
    os.system("streamlit run src/app/app.py")