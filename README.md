# LLM REST API

The simplest possible Python code for running local LLM inference as a REST API server (with a simple client).

This package lets you start an inference server for Hugging Face–compatible models (like LLaMA, Qwen, GPT-OSS, etc.) on your own computer or server, and make it accessible to applications via HTTP.

---

## Installation

From PyPI (recommended):

```bash
pip install min-llm-server-client
```

From source:

```bash
git clone https://github.com/afshinsadeghi/min_llm_server_client.git
cd min_llm_server_client
pip install .
```

---

## Usage

### Starting the Server

After installation, you can launch the server with the provided CLI entrypoint:

```bash
min-llm-server --model_name meta-llama/Llama-3.3-70B-Instruct --max_new_tokens 100 --device cuda:0
```

Options:
- `--model_name` : Hugging Face model name or local path (e.g. `openai/gpt-oss-20b`, `meta-llama/Llama-3.3-70B-Instruct`, or `/path/to/model`).
- `--max_new_tokens` : maximum number of tokens to generate in response.
- `--device` : `cpu`, `cuda:0`, `cuda:1`, etc.

Example (CPU run):

```bash
min-llm-server --model_name openai/gpt-oss-20b --max_new_tokens 50 --device cpu
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

### Performance notes

- Running **LLaMA 3.1 8B**:
  - Intel CPU → ~30 seconds per request, ~2.4 GB RAM
  - A100 GPU → <1 second per request, ~34 GB GPU memory, ~4.8 GB CPU RAM

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

This project is open source under the [Apache 2.0 License](./LICENSE).

---

## Author
Afshin Sadeghi  
📧 sadeghi.afshin@gmail.com  
🔗 [GitHub](https://github.com/afshinsadeghi)  
🔗 [Google Scholar](https://scholar.google.com/citations?user=uWTszVEAAAAJ&hl=en&oi=ao)  
🔗 [LinkedIn](https://www.linkedin.com/in/afshin-sadeghi)