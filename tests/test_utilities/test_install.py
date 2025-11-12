#!/usr/bin/env python3
import sys
import subprocess


def install_library(library_name, package_name=None):
    """Install a missing library using pip."""
    if package_name is None:
        package_name = library_name

    try:
        # Use subprocess to run pip install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return True, f"Successfully installed {package_name}"
        else:
            return False, f"Failed to install {package_name}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"Timeout installing {package_name}"
    except Exception as e:
        return False, f"Error installing {package_name}: {str(e)}"


# Test installing matplotlib
print("Testing matplotlib installation...")
try:
    import matplotlib

    print("✓ matplotlib already available")
except ImportError:
    success, message = install_library("matplotlib")
    print(message)
    if success:
        try:
            import matplotlib

            print("✓ matplotlib successfully installed and imported")
        except ImportError:
            print("✗ matplotlib installation failed")

# Test installing seaborn
print("\nTesting seaborn installation...")
try:
    import seaborn

    print("✓ seaborn already available")
except ImportError:
    success, message = install_library("seaborn")
    print(message)
    if success:
        try:
            import seaborn

            print("✓ seaborn successfully installed and imported")
        except ImportError:
            print("✗ seaborn installation failed")
