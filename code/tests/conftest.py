# conftest.py — pytest configuration
import sys
from pathlib import Path

# Ensure the code directory is on the path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))
