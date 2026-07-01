from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 로컬 임베딩 모델 지정 (OpenAI API를 사용하지 않고 로컬 모델을 사용)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("../data").load_data()
print(f"loaded {len(documents)} chunks")

index = VectorStoreIndex.from_documents(documents)
print("index built successfully")

print(index.vector_store.to_dict().keys())