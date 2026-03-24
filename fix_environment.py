import sys
import subprocess
import os

def check_package(package_name):
    """Checks if a package is available for import."""
    # Mapping of package names to import names
    import_mapping = {
        'scikit-learn': 'sklearn',
        'imbalanced-learn': 'imblearn'
    }
    import_name = import_mapping.get(package_name, package_name)
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def install_requirements():
    print("\n[INFO] Attempting to install dependencies from requirements.txt...")
    try:
        # Using sys.executable to ensure we use the same interpreter as the script
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[SUCCESS] Installation complete!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] During installation: {e}")

def main():
    print("=" * 50)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 50)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version:    {sys.version.split('|')[0].strip()}")
    print("-" * 50)
    print("Current sys.path:")
    for path in sys.path:
        print(f"  - {path}")
    print("-" * 50)
    
    # Full list from requirements.txt
    required_packages = [
        'flask', 'scapy', 'pandas', 'numpy', 'xgboost', 
        'scikit-learn', 'matplotlib', 'joblib', 'shap', 
        'tensorflow', 'imbalanced-learn'
    ]
    
    missing = []
    print("Checking packages:")
    for pkg in required_packages:
        if check_package(pkg):
            print(f"  [OK]      {pkg}")
        else:
            print(f"  [MISSING] {pkg}")
            missing.append(pkg)
            
    if missing:
        print(f"\n[!] {len(missing)} packages are missing. Let's try to fix them.")
        install_requirements()
        
        # Re-check after installation
        print("\nRe-checking packages after installation:")
        still_missing = []
        for pkg in missing:
            if check_package(pkg):
                print(f"  [FIXED]   {pkg}")
            else:
                print(f"  [STILL MISSING] {pkg}")
                still_missing.append(pkg)
        
        if still_missing:
            print(f"\n[WARNING] Some packages could not be installed: {', '.join(still_missing)}")
    else:
        print("\n[CONGRATS] All packages are correctly installed in this environment.")

    print("\n" + "=" * 50)
    print("IDE SETUP TIP")
    print("=" * 50)
    print("If your IDE still shows import errors:")
    print("1. Open VS Code Command Palette (Ctrl+Shift+P).")
    print("2. Type 'Python: Select Interpreter'.")
    print(f"3. Select the path below to match this environment:")
    print(f"   >>> {sys.executable} <<<")
    print("=" * 50)

if __name__ == "__main__":
    main()
