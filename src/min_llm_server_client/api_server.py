# filepath: /llm_server_client/src/local_llm_inference_server_api.py
import os
import sentencepiece as spm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from flask import Flask, request, jsonify
import argparse
import json
import torch
import pynvml

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() in self.stop_ids

class ModelRunner():
    def __init__(self, setting):
        self.device = setting.device
        self.max_new_tokens = setting.max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=setting.llm_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # IMPORTANT:
        # - device_map="auto" enables sharding.
        # - max_memory forces the placer to spread across visible GPUs.
        # - torch_dtype=bfloat16 (or float16) avoids fp32 OOM.
        #device_map_ = None
        if setting.device != "auto" and setting.device != "cpu":

            # strip "cuda:" prefix if present
            dev_str = setting.device.replace("cuda:", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
            print("Set CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            gpus = self.pick_gpus(min_free_gib=50)
            if gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            else:
                # fall back to CPU-only or raise
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

                # Diagnostics
            print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
            print("torch.cuda.device_count():", torch.cuda.device_count())

            #if torch.cuda.device_count() > 1:
            #    print("Multiple GPUs detected; setting device_map='auto'")
            #    #device_map_ = "auto"
            #    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
            #    print("Set CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])

        #    device_map=_ "auto"
        #    else: 

            
        self.model = AutoModelForCausalLM.from_pretrained(
            setting.llm_path,
            device_map= "auto",
            torch_dtype=torch.bfloat16,         # prefer bf16; use float16 if needed
            low_cpu_mem_usage=True
        )

        print("hf_device_map:", getattr(self.model, "hf_device_map", None))


    def pick_gpus(self, min_free_gib=12, top_k=None):
        pynvml.nvmlInit()
        stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            free_gib = mem.free / (1024**3)
            stats.append((i, free_gib))
        pynvml.nvmlShutdown()
        # keep GPUs with enough free mem
        candidates = [i for i, free in stats if free >= min_free_gib]
        # optionally keep only top_k by free memory
        if top_k:
            candidates = [i for i, _ in sorted(stats, key=lambda x: x[1], reverse=True)[:top_k]]
        return candidates

    def run_query(self, query):
        # Do NOT pin inputs to a specific GPU with a sharded model.
        inputs = self.tokenizer(
            [query],
            return_token_type_ids=False,
            padding=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,                    # leave on CPU; HF routes to shards
                do_sample=False, #True,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria = stop_criteria,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_length = inputs.input_ids.shape[1]
        result = self.tokenizer.batch_decode(
            generated_ids[:, input_length:],
            skip_special_tokens=True
        )[0]
        return result


app = Flask(__name__)

@app.route('/llm/q', methods=['POST'])
def read_question():
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
    import argparse

    setting = SimpleNamespace()
    parser = argparse.ArgumentParser("Minimal LLM Server")
    parser.add_argument("--model_name", default="openai/gpt-oss-20b", help="default is openai/gpt-oss-20b you can use other models such as meta-llama/Llama-3.3-70B-Instruct", type=str)
    parser.add_argument("--max_new_tokens", default=500, type=int)
    parser.add_argument("--device", default="", type=str, help="cpu | auto | cuda:0 etc.")
    args = parser.parse_args()

    setting.llm_path = args.model_name
    setting.max_new_tokens = args.max_new_tokens
    setting.device = args.device

    global llm_runner
    llm_runner = ModelRunner(setting)
    global stop_criteria
    stop_criteria = StoppingCriteriaList([StopOnTokens([tokenizer.eos_token_id])])


    print("Starting the server with model:", setting.llm_path)
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
