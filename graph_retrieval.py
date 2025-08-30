from openai import OpenAI

base_url ="http://0.0.0.0:10085/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

message = [
    {
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the knowledge graph.",
    },
    {
        "role": "user",
        "content": "Question: How many module problems are there?",
    }
]
response = client.chat.completions.create(
    model="llama",
    messages=message,
    max_tokens=2048,
    temperature=0.5
)
print(response.choices[0].message.content)
