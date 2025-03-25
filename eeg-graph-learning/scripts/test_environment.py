# scripts/test_environment.py
import sys
import importlib
import pkg_resources

def check_package(package_name):
    """Check if a package is installed and print its version."""
    try:
        package = importlib.import_module(package_name)
        if hasattr(package, '__version__'):
            version = package.__version__
        else:
            try:
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "Unknown version"
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def main():
    """Check if all required packages are installed."""
    packages = [
        "numpy", 
        "scipy", 
        "matplotlib", 
        "pandas", 
        "torch", 
        "torch_geometric", 
        "sklearn", 
        "mne", 
        "tensorboard", 
        "pytest",
        "optuna"
    ]
    
    print("Testing Python environment...")
    print(f"Python version: {sys.version}")
    
    all_installed = True
    for package in packages:
        if not check_package(package):
            all_installed = False
    
    # Test eeglearn configuration
    try:
        from eeglearn.config import Config
        print(f"✅ eeglearn.config: Random seed = {Config.RANDOM_SEED}, Deterministic = {Config.DETERMINISTIC}")
    except ImportError:
        print("❌ eeglearn.config: Not properly set up")
        all_installed = False
    
    # Test GPU availability for PyTorch
    if check_package("torch"):
        import torch
        
        # Check for NVIDIA GPU
        cuda_available = torch.cuda.is_available()
        print(f"\nNVIDIA GPU available: {cuda_available}")
        if cuda_available:
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        # Check for Apple Silicon MPS (Metal Performance Shaders)
        mps_available = False
        if hasattr(torch.backends, "mps"):
            mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            print(f"Apple MPS available: {mps_available}")
            if mps_available:
                print("Using Apple Silicon GPU acceleration")
        
        # Summary of compute availability
        if cuda_available or mps_available:
            print("✅ GPU acceleration is available!")
        else:
            print("⚠️ No GPU acceleration found. Using CPU only.")
    
    if all_installed:
        print("\n✅ All packages are installed correctly!")
    else:
        print("\n❌ Some packages are missing. Please check your environment setup.")

if __name__ == "__main__":
    main()
