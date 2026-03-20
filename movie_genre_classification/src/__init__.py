"""
src/__init__.py
~~~~~~~~~~~~~~~
Ensures the project root is on sys.path so that `from src.X import Y`
works regardless of the working directory from which Python is invoked.
"""
import os
import sys

# Insert the project root (parent of this file's directory) at the front
# of sys.path if it isn't already there.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
