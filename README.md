


# Minimal LLM Server, for API calls  


The simplest possible Python code for running local LLM inference as a REST API server (with a simple client).

This package lets you start an inference server for Hugging Face–compatible models (like LLaMA, Qwen, GPT-OSS, etc.) on your own computer or server, and make it accessible to applications via HTTP.

**NEW:** Now supports both standard HuggingFace Transformers and high-performance vLLM backends!

See the [Tutorial](https://medium.com/@sadeghi.afshin/run-gpt-oss-20b-and-gpt-oss-120b-locally-with-a-minimal-api-server-in-the-style-of-openai-1872e68a93b7) page for extented info.
---

## Backend Options

This package now supports two inference backends:

### 1. **HuggingFace Transformers (Standard)**
- ✓ Widely compatible
- ✓ CPU support available
- ✓ Smaller installation size
- ✓ Good for development and testing

### 2. **vLLM Optimized (High-Performance)** 🚀
- ✓ Up to **24x faster** throughput than standard transformers
- ✓ Lower latency for single requests
- ✓ Better GPU memory utilization with PagedAttention
- ✓ Automatic multi-GPU support with tensor parallelism
- ✓ Continuous batching for higher throughput
- ⚠ Requires CUDA GPUs (no CPU support)
- ⚠ Best for production deployments

---

## Installation

## Installation by pip ![PyPl Total Downloads](https://img.shields.io/pepy/dt/min_llm_server_client)

**Standard Installation (HuggingFace):**

```bash
pip install min-llm-server-client
```

**With vLLM Support:**

```bash
pip install "min-llm-server-client[vllm]"
```

#### Option 2: Installation From Source:

```bash
git clone https://github.com/afshinsadeghi/min_llm_server_client.git
cd min_llm_server_client

# Standard installation
pip install .

# Or with vLLM support
pip install ".[vllm]"
```

---

## Usage

### Starting the Server

#### Standard Server (HuggingFace Transformers)

```bash
min-llm-server --model_name meta-llama/Llama-3.3-70B-Instruct --max_new_tokens 100 --device cuda:0
```

#### #### vLLM Optimized Server (High-Performance) 🚀

```bash
min-llm-server-vllm --model_name meta-llama/Llama-3.3-70B-Instruct --max_new_tokens 100 --device auto
```

**Command Options:**
- `--model_name` : Hugging Face model name or local path
   suggested models:
    `openai/gpt-oss-20b`
    `openai/gpt-oss-120b`
    `meta-llama/Llama-3.3-70B-Instruct`
    `meta-llama/Llama-3.1-8B`
    `Qwen/Qwen3-0.6B`
    `Qwen/Qwen2-VL-72B-Instruct-AWQ`
    `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
  
    or it can use a local model on your device with `/path/to/model`.

- `--max_new_tokens` : maximum number of tokens to generate in response.
- `--device` : Device selection 
  - `auto`  - Auto-detect available GPUs (default)
  - `cpu`,  - Force CPU (HuggingFace only, vLLM requires GPU)
  -    `cuda:0`, `cuda:1` , or a list of GPU cores: `cuda:2,3,4,5,6,7`. 

If the device parameter is not given or is `auto`, it finds the available GPU cores and uses them and if no gpu is available, it uses CPU instead. 

#### Example run: 

Standard server with default settings (auto GPU detection):
```bash
min-llm-server 
```

Standard server on a specific GPU (e.g., GPU 0):
```bash
min-llm-server --model_name openai/gpt-oss-20b --device cuda:0
```

Standard server on a specific GPU (e.g., GPU 1):
```bash
min-llm-server --model_name openai/gpt-oss-120b --device cuda:1
```

Standard server forced on CPU:
```bash
min-llm-server --model_name openai/gpt-oss-20b --max_new_tokens 50 --device cpu
```

vLLM server with auto GPU detection (uses all available GPUs):
```bash
min-llm-server-vllm --model_name meta-llama/Llama-3.3-70B-Instruct
```

vLLM server on a specific GPU (e.g., GPU 2):
```bash
min-llm-server-vllm --model_name meta-llama/Llama-3.3-70B-Instruct --device cuda:2
```

Standard server on a several GPUs:
```bash

min-llm-server --model_name meta-llama/Llama-3.3-70B-Instruct --device cuda:2,3,4,5,6,7

```
---

### Sending Queries

Once the server is running (default: `http://127.0.0.1:5000/llm/q`), you can query it with `curl` or Python.

**Curl:**

```bash
curl -X POST http://127.0.0.1:5000/llm/q \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Earth?", "key": "key1"}'
```

**Python client:**

```python
from min_llm_server_client.local_llm_inference_api_client import send_query

response = send_query("What is the capital of France?", user="user1", key="key1")
print(response)
```

---

### Performance Comparison

**LLaMA 3.1 8B - Standard HuggingFace Backend:**
- Intel CPU → ~30 seconds per request, ~2.4 GB RAM
- A100 GPU → <1 second per request, ~34 GB GPU memory, ~4.8 GB CPU RAM

**LLaMA 3.1 8B - vLLM Optimized Backend:**
- A100 GPU → ~0.1-0.3 seconds per request (3-10x faster)
- Better memory efficiency with PagedAttention
- Supports higher concurrent request throughput

**Performance Tips:**
- Use vLLM for production deployments with high request volumes
- Use standard backend for development, testing, or CPU-only environments
- vLLM automatically utilizes multiple GPUs with tensor parallelism
- Both backends support the same API, making it easy to switch

---

## Project Structure

```
min_llm_server_client/
├── src/
│   ├── local_llm_inference_api_client.py
│   ├── local_llm_inference_server_api.py
│   └── ...
└── README.md
```

---

## License

This project is open source under the [Apache 2.0 License](./LICENSE-2.0.txt).

---

## Author
Afshin Sadeghi   
🔗 [GitHub](https://github.com/afshinsadeghi)  
🔗 [Google Scholar](https://scholar.google.com/citations?user=uWTszVEAAAAJ&hl=en&oi=ao)  
🔗 [LinkedIn](https://www.linkedin.com/in/afshin-sadeghi)
