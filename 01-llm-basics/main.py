import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("AI Assistant (type 'exit' to quit)\n")

# System: Here’s how you should behave
# User: Here’s what I’m asking

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are a senior healthcare AI architect. Be precise, structured, and avoid fluff."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        )

    except Exception as e:
        print ("Error", e)

print("\nAI:", response.output_text)
print("-" * 40)

with open("chat_log.txt", "a") as f:
    f.write(f"User: {user_input}\n")
    f.write(f"AI: {response.output_text}\n\n")