import torch
import sys
import platform

def check_cuda_installation():
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Test CUDA functionality
        print("\n=== Testing CUDA Functionality ===")
        x = torch.rand(5, 3)
        print("CPU tensor:", x)
        x = x.cuda()
        print("GPU tensor:", x)
        print("CUDA test successful!")
    else:
        print("\nCUDA is not available. Please check:")
        print("1. NVIDIA GPU is installed")
        print("2. NVIDIA drivers are installed")
        print("3. CUDA toolkit is installed")
        print("4. PyTorch is installed with CUDA support")

if __name__ == "__main__":
    check_cuda_installation() 