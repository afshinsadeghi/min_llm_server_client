### LLM REST API

 The simplest possible python code for inference call of LLMs as a REST API server and a simple client for it
 
 This setting  both the server and client being written in Python and running on the same computer.

 This is the basic code if you want to call LLMs in your server, or own computer and make it accessible to applications and codes locally.




#### Installation

To install the package, clone the repository and run the following command in the project root directory:

```
pip install .
```
You can install the dependencies using:

```
pip install -r requirements.txt
```



## Usage


### Configuration

The configuration settings, such as the model path and device settings, can be found in `setting` parameter of server. Make sure to update these settings according to your environment.


### Starting the Server

To start the LLM inference server, run the following command:



### To run:

```
 python -m min_llm_server_client.src.local_llm_inference_server_api
```
You can set the arguments such as:  
-  --model_name  
- --max_new_tokens  
- --device  

For example:  
```
 python -m min_llm_server_client.src.local_llm_inference_server_api --model_name  meta-llama/Llama-3.3-70B-Instruct --max_new_tokens  100 --device cuda:1
```
device could be cpu or 0 or 1 or any other number meaning the core gpu number to use

running on cpu:
```
 python -m min_llm_server_client.src.local_llm_inference_server_api --model_name  openai/gpt-oss-20b --max_new_tokens  100 --device cpu
```

##### Usage on browser:

 get test with curl http://127.0.0.1:5000/llm/q

or: 
post test no user : curl -X POST http://127.0.0.1:5000/llm/q  -H "Content-Type: application/json" -d '{"query": "what is earth?"}'

post test no user : curl -X POST http://127.0.0.1:5000/llm/q  -H "Content-Type: application/json" -d '{"query": "what is earth?" , "key": "key1"}'


Local test runs using lamma 3.1 8B:

intel cpu takes 30 seonds : memory cpu 2.4 used GB
A100 gpu less than a seoncd; memroy GPU 34 GB , cpu  4.8 GB

#### Author's contact : 
```
sadeghi.afshin@gmail.com
```

## License

This project is open source. licensed under the Apache 2.0 License. See the LICENSE file for more details.


#### Explnation:
This project provides a simple REST API server and client for interacting with a local language model (LLM) inference server. The server is built using Flask and allows users to send queries to the model and receive generated responses.

##### Project Structure

```
llm_server_client
├── src
│   ├── __init__.py
│   ├── local_llm_inference_api_client.py
│   ├── local_llm_inference_server_api.py
│   └── setting.py
├── setup.py
└── README.md
```


#### Using in third party code 

Sending Queries:  

To interact with the server, you can use the client provided in `src/local_llm_inference_api_client.py`. This client includes functions to send queries to the server and handle responses.

Example usage 

Here is a simple example of how to send a query to the server:

```python
from src.local_llm_inference_api_client import send_query

response = send_query("What is the capital of France?", user="user1", key="key1")
print(response)
```

#### Dependencies

This project requires the following Python packages:

- Flask
- transformers
- sentencepiece

