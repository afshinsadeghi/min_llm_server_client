# filepath: /llm_server_client/src/min_llm_server_client/api_server_vllm.py
"""
Minimal LLM Server API - vLLM Optimized Version

This server provides a high-performance API for running inference with large language models
using vLLM (https://github.com/vllm-project/vllm), which offers significant speed improvements
over standard HuggingFace transformers through:
- PagedAttention for efficient memory management
- Continuous batching for higher throughput
- Optimized CUDA kernels
- Tensor parallelism for multi-GPU setups

Performance benefits:
- Up to 24x higher throughput than HuggingFace Transformers
- Lower latency for single requests
- Better GPU memory utilization
- Automatic multi-GPU support with tensor parallelism

Important implementation notes:
- vLLM automatically handles GPU detection and tensor parallelism
- Supports both single-GPU and multi-GPU configurations
- Uses OpenAI-compatible sampling parameters
- Efficient batching even for single requests
"""

import os
from flask import Flask, request, jsonify
import argparse

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    import torch
    import pynvml
except ImportError as e:
    VLLM_AVAILABLE = False
    print(f"WARNING: vLLM or its dependencies are not installed. Error: {e}")
    print("Please install it with: pip install vllm")
    # Import torch and pynvml anyway for error handling
    try:
        import torch
        import pynvml
    except ImportError:
        pass


class VLLMModelRunner():
    """
    Model runner using vLLM for optimized inference.
    
    vLLM provides significant performance improvements over standard transformers:
    - PagedAttention: Efficient KV cache management
    - Continuous batching: Process requests as they arrive
    - Optimized kernels: Faster attention and sampling
    - Tensor parallelism: Automatic multi-GPU support
    """
    
    def __init__(self, setting):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install it with:\n"
                "pip install vllm\n"
                "or for CUDA 11.8:\n"
                "pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118"
            )
        
        self.max_new_tokens = setting.max_new_tokens
        
        # Configure GPU visibility if specific device is requested
        if setting.device != "auto" and setting.device != "cpu":
            # strip "cuda:" prefix if present
            dev_str = setting.device.replace("cuda:", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
            print(f"Set CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            tensor_parallel_size = 1
        else:
            # Auto-detect GPUs with sufficient memory
            gpus = self.pick_gpus(min_free_gib=50)
            if gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                tensor_parallel_size = len(gpus)
                print(f"Auto-detected {len(gpus)} GPUs: {gpus}")
            else:
                # Fall back to CPU (vLLM doesn't support CPU, so this will fail gracefully)
                print("WARNING: No suitable GPUs found. vLLM requires CUDA GPUs.")
                tensor_parallel_size = 1
        
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        
        # Initialize vLLM model
        # vLLM automatically handles:
        # - Model loading and sharding across GPUs
        # - KV cache allocation
        # - Attention optimization
        # - Batching and scheduling
        print(f"Loading model with vLLM: {setting.llm_path}")
        self.model = LLM(
            model=setting.llm_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",  # Use bfloat16 for better performance
            trust_remote_code=True,
            max_model_len=None,  # Auto-detect based on model config
            gpu_memory_utilization=0.90,  # Use 90% of GPU memory for KV cache
        )
        
        print("vLLM model loaded successfully!")
        
    def pick_gpus(self, min_free_gib=12, top_k=None):
        """
        Select GPUs with sufficient free memory.
        
        Args:
            min_free_gib: Minimum free memory in GiB required
            top_k: Optional limit on number of GPUs to use
            
        Returns:
            List of GPU indices with sufficient memory
        """
        pynvml.nvmlInit()
        stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            free_gib = mem.free / (1024**3)
            stats.append((i, free_gib))
        pynvml.nvmlShutdown()
        
        # Keep GPUs with enough free memory
        candidates = [i for i, free in stats if free >= min_free_gib]
        
        # Optionally keep only top_k by free memory
        if top_k:
            candidates = [i for i, _ in sorted(stats, key=lambda x: x[1], reverse=True)[:top_k]]
        
        return candidates
    
    def run_query(self, query):
        """
        Run inference on a query using vLLM.
        
        vLLM uses SamplingParams to control generation, similar to HuggingFace
        but with optimized implementation.
        
        Args:
            query: Input text prompt
            
        Returns:
            Generated text response
        """
        # Configure sampling parameters
        # These are similar to HuggingFace but optimized for vLLM
        sampling_params = SamplingParams(
            temperature=0.1,           # Lower = more deterministic
            top_p=0.9,                 # Nucleus sampling
            max_tokens=self.max_new_tokens,
            repetition_penalty=1.2,
            # vLLM uses different parameter names for some settings
            # best_of=2 is similar to num_beams in HuggingFace
            # but uses sampling instead of beam search
        )
        
        # Generate response
        # vLLM automatically handles batching, even for single requests
        outputs = self.model.generate([query], sampling_params)
        
        # Extract the generated text
        # vLLM returns a list of RequestOutput objects
        result = outputs[0].outputs[0].text
        
        return result


app = Flask(__name__)

@app.route('/llm/q', methods=['POST'])
def read_question():
    """
    API endpoint for LLM queries.
    
    Expects JSON with:
    - query: The text prompt
    - key: API key for authentication
    
    Returns:
    - JSON with the generated response
    """
    rq = request.get_json()
    question = rq.get('query', "")
    key = rq.get('key', "no key is provided")
    
    if key == "key1":
        result = llm_runner.run_query(question)
    else:
        result = "user key is unknown"
        
    return jsonify(message="Success", statusCode=200, query=question, answer=result), 200


def main():
    from types import SimpleNamespace
    
    setting = SimpleNamespace()
    parser = argparse.ArgumentParser(
        "Minimal LLM Server - vLLM Optimized",
        description="High-performance LLM inference server using vLLM"
    )
    parser.add_argument(
        "--model_name", 
        default="openai/gpt-oss-20b", 
        help="HuggingFace model name or local path (default: openai/gpt-oss-20b). "
             "Examples: meta-llama/Llama-3.3-70B-Instruct, Qwen/Qwen2-VL-72B-Instruct-AWQ",
        type=str
    )
    parser.add_argument(
        "--key", 
        default="key1", 
        type=str, 
        help="Simple API key for server access"
    )
    parser.add_argument(
        "--max_new_tokens", 
        default=500, 
        type=int,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        type=str, 
        help="Device selection: 'auto' (auto-detect GPUs), 'cuda:0', 'cuda:1', etc. "
             "Note: vLLM requires CUDA GPUs and does not support CPU inference"
    )
    args = parser.parse_args()
    
    setting.key = args.key
    setting.llm_path = args.model_name
    setting.max_new_tokens = args.max_new_tokens
    setting.device = args.device
    
    global llm_runner
    llm_runner = VLLMModelRunner(setting)
    
    print("\n" + "="*60)
    print("vLLM-Optimized LLM Server Starting")
    print("="*60)
    print(f"Model: {setting.llm_path}")
    print(f"Max tokens: {setting.max_new_tokens}")
    print(f"Device: {setting.device}")
    print("="*60 + "\n")
    
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
