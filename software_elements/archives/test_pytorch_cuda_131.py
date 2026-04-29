"""
Test script to check PyTorch CUDA 13.1 compatibility.
This script verifies if the installed PyTorch version supports CUDA 13.1.
"""

import torch
import sys
from packaging import version

def test_pytorch_cuda_compatibility():
    """
    Test PyTorch CUDA 13.1 compatibility.
    """
    print("=" * 60)
    print("PyTorch CUDA 13.1 Compatibility Test")
    print("=" * 60)
    
    # Get PyTorch version
    pytorch_version = torch.__version__
    print(f"\n1. PyTorch Version: {pytorch_version}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"\n2. CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version PyTorch was compiled with
        cuda_version = torch.version.cuda
        print(f"3. CUDA Version (compiled with PyTorch): {cuda_version}")
        
        # Get current GPU info
        print(f"\n4. GPU Information:")
        print(f"   - GPU Count: {torch.cuda.device_count()}")
        print(f"   - Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("3. CUDA Version: Not available (CUDA is not available)")
    
    # Check cuDNN version if available
    try:
        cudnn_version = torch.backends.cudnn.version()
        print(f"\n5. cuDNN Version: {cudnn_version}")
    except Exception as e:
        print(f"\n5. cuDNN Version: Not available ({str(e)})")
    
    # Test basic CUDA operation
    print(f"\n6. Testing Basic CUDA Operations:")
    try:
        if cuda_available:
            # Create a tensor on GPU
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.matmul(x, y)
            print(f"   ✓ Matrix multiplication on GPU: SUCCESS")
            print(f"   - Result shape: {z.shape}")
            print(f"   - Result device: {z.device}")
        else:
            print("   ✗ CUDA not available - cannot test GPU operations")
    except Exception as e:
        print(f"   ✗ CUDA operation failed: {str(e)}")
    
    # Check compatibility with CUDA 13.1
    print(f"\n7. CUDA 13.1 Compatibility Check:")
    if cuda_available and cuda_version:
        try:
            # Parse versions for comparison
            # CUDA 13.1 compatibility: PyTorch typically supports CUDA versions
            # within a reasonable range
            cuda_ver = version.parse(cuda_version)
            target_cuda = version.parse("13.1")
            
            # PyTorch usually supports ±1 minor version variations
            min_supported = version.parse("13.0")
            max_supported = version.parse("13.2")
            
            if min_supported <= cuda_ver <= max_supported:
                print(f"   ✓ PyTorch {pytorch_version} is COMPATIBLE with CUDA 13.1")
                print(f"   - Compiled with CUDA {cuda_version}")
            else:
                print(f"   ⚠ PyTorch {pytorch_version} compiled with CUDA {cuda_version}")
                print(f"   - This may not be fully compatible with CUDA 13.1")
                print(f"   - For guaranteed CUDA 13.1 support, look for PyTorch versions")
                print(f"     compiled with CUDA 13.1 or 13.x")
        except Exception as e:
            print(f"   ⚠ Could not parse versions: {str(e)}")
    else:
        print("   ✗ CUDA not available - cannot determine compatibility")
    
    # Recommendations
    print(f"\n8. Recommendations:")
    print(f"   If you need CUDA 13.1 support:")
    print(f"   - Install PyTorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131")
    print(f"   - Or use conda: conda install pytorch::pytorch torchvision torchaudio pytorch-cuda=13.1 -c pytorch -c nvidia")
    
    print(f"\n" + "=" * 60)

if __name__ == "__main__":
    test_pytorch_cuda_compatibility()
