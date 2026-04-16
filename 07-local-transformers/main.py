from transformers import pipeline


classifier = pipeline("sentiment-analysis")

texts = [
    "I love how quickly this system works.",
    "This environment is stressful and frustrating.",
    "The results are mixed and uncertain."
]
   


results = classifier(texts)

for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")
    print("-" * 40)

generator = pipeline("text-generation", model="distilgpt2")

prompt = "Healthcare AI systems need"

result = generator(prompt, max_length=40, num_return_sequences=1)

print(result[0]["generated_text"])
