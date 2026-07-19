from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "models/tokenizer/all-MiniLM-L6-v2",
    local_files_only=True
)

text = "This is a test for our RAG pipeline."

tokens = tokenizer.encode(text, add_special_tokens=False)

print(tokens)

print(tokenizer.decode(tokens))