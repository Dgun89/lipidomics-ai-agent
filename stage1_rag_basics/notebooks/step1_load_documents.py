from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("../data").load_data()

print(f"Loaded {len(documents)} chunks")
print(documents[0])
