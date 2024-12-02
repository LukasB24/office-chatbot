from ollama import Client

client = Client()

response = client.generate(
    prompt="Generate a creative haiku",
    model="llama3.1:8b",
    options={"temperature":0}
)

print(response['response'])