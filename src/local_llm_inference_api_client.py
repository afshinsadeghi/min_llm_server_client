import requests

class LLMClient:
    def __init__(self, base_url, user_key):
        self.base_url = base_url
        self.user_key = user_key

    def ask_question(self, query):
        payload = {
            'query': query,
            'user': 'client_user',
            'key': self.user_key
        }
        response = requests.post(f"{self.base_url}/llm/q", json=payload)
        if response.status_code == 200:
            return response.json().get('answer', 'No answer found.')
        else:
            return f"Error: {response.status_code} - {response.text}"

if __name__ == "__main__":
    client = LLMClient(base_url="http://127.0.0.1:5000", user_key="key1")
    question = "What is the capital of France?"
    answer = client.ask_question(question)
    print(f"Question: {question}\nAnswer: {answer}")