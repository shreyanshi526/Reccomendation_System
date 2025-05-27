import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Running Python App in Dev Mode...")
os.system("python mentor_ai/main.py")
