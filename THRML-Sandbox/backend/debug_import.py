import sys
import os
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
thrml_path = os.path.join(current_dir, '..', 'thrml-main')
print(f"Adding path: {thrml_path}")
sys.path.append(thrml_path)

print(f"Sys Path: {sys.path}")

try:
    import thrml
    print("Import successful!")
    print(f"THRML file: {thrml.__file__}")
except ImportError:
    print("Import failed!")
    traceback.print_exc()
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
