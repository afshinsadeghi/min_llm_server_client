#!/usr/bin/env python3
"""
Interactive installation script for min_llm_server_client

This script allows users to choose between:
1. Standard HuggingFace Transformers backend (default)
2. vLLM optimized backend (high-performance)
"""

import subprocess
import sys
import os


def print_banner():
    """Print installation banner"""
    print("\n" + "="*70)
    print("  Minimal LLM Server - Installation")
    print("="*70)
    print()


def print_comparison():
    """Print comparison between backends"""
    print("\nBackend Comparison:")
    print("-" * 70)
    print("\n1. HuggingFace Transformers (Standard)")
    print("   ✓ Widely compatible")
    print("   ✓ CPU support available")
    print("   ✓ Smaller installation size")
    print("   ✓ Good for development and testing")
    print("   - Slower inference speed")
    print("   - Higher memory usage")
    print()
    print("2. vLLM Optimized (High-Performance)")
    print("   ✓ Up to 24x faster throughput")
    print("   ✓ Lower latency")
    print("   ✓ Better GPU memory utilization")
    print("   ✓ Automatic multi-GPU support")
    print("   ✓ PagedAttention for efficient KV cache")
    print("   - Requires CUDA GPUs (no CPU support)")
    print("   - Larger installation size")
    print("   - Best for production deployments")
    print()
    print("-" * 70)


def install_package(with_vllm=False):
    """Install the package with optional vLLM support"""
    print("\n📦 Installing min_llm_server_client package...")
    try:
        if with_vllm:
            print("Installing with vLLM support (this may take a few minutes)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e", ".[vllm]"
            ])
        else:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-e", "."
            ])
        print("✓ Package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing package: {e}")
        if with_vllm:
            print("\nIf you encounter CUDA version issues with vLLM, try:")
            print("  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118")
        return False


def create_config_file(backend):
    """Create a configuration file to remember the chosen backend"""
    config_dir = os.path.expanduser("~/.min_llm_server")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.txt")
    
    with open(config_file, "w") as f:
        f.write(f"backend={backend}\n")
    
    print(f"\n✓ Configuration saved to {config_file}")


def print_usage_instructions(backend):
    """Print usage instructions based on chosen backend"""
    print("\n" + "="*70)
    print("  Installation Complete!")
    print("="*70)
    
    if backend == "vllm":
        print("\n🚀 You can now start the vLLM-optimized server with:")
        print("\n  python -m min_llm_server_client.api_server_vllm --model_name <model>")
        print("\nOr use the standard server:")
        print("\n  min-llm-server --model_name <model>")
    else:
        print("\n🚀 You can now start the server with:")
        print("\n  min-llm-server --model_name <model>")
        print("\nExample:")
        print("\n  min-llm-server --model_name meta-llama/Llama-3.3-70B-Instruct")
    
    print("\n" + "="*70)
    print()


def main():
    """Main installation flow"""
    print_banner()
    print_comparison()
    
    # Ask user for backend choice
    while True:
        print("\nWhich backend would you like to install?")
        print("  1. HuggingFace Transformers (Standard) - Recommended for most users")
        print("  2. vLLM Optimized (High-Performance) - Recommended for production")
        print("  3. Both (install both backends)")
        
        choice = input("\nEnter your choice (1/2/3) [default: 1]: ").strip()
        
        if choice == "" or choice == "1":
            backend = "huggingface"
            install_vllm_flag = False
            break
        elif choice == "2":
            backend = "vllm"
            install_vllm_flag = True
            break
        elif choice == "3":
            backend = "both"
            install_vllm_flag = True
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print(f"\n✓ Selected backend: {backend}")
    
    # Install the package with optional vLLM support
    if not install_package(with_vllm=install_vllm_flag):
        print("\n✗ Installation failed!")
        if install_vllm_flag:
            print("\n⚠ Trying to install without vLLM...")
            if not install_package(with_vllm=False):
                print("\n✗ Installation failed completely!")
                sys.exit(1)
            else:
                print("\n⚠ Installed successfully without vLLM support")
                backend = "huggingface"
        else:
            sys.exit(1)
    
    # Save configuration
    create_config_file(backend)
    
    # Print usage instructions
    print_usage_instructions(backend)


if __name__ == "__main__":
    main()
