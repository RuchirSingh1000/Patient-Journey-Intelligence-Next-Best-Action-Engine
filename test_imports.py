"""
Simple test to check if imports work
"""
print("Testing imports...")

try:
    import pandas as pd
    print("[OK] pandas imported")
except Exception as e:
    print(f"[FAIL] pandas import failed: {e}")

try:
    import numpy as np
    print("[OK] numpy imported")
except Exception as e:
    print(f"[FAIL] numpy import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("[OK] matplotlib imported")
except Exception as e:
    print(f"[FAIL] matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("[OK] seaborn imported")
except Exception as e:
    print(f"[FAIL] seaborn import failed: {e}")

try:
    from datetime import datetime
    print("[OK] datetime imported")
except Exception as e:
    print(f"[FAIL] datetime import failed: {e}")

print("Import test complete.")