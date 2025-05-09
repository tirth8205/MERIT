# Initialize the adapter
adapter = HuggingFaceAdapter(
    api_token=api_token,
    model_name="meta-llama/Meta-Llama-3-8B-Instruct"  # Free to use model
)

# Generate text
prompt = "Evaluate the following logical argument: All humans are mortal. Socrates is human. Therefore, Socrates is mortal."
response = adapter.generate(prompt)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
