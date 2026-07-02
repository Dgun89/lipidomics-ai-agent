import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# question = "What lipid classes are discussed in this paper?"
# question = "What metrics did MetaBench use to evaluate LLM performance?"
# question = "What is the accuracy of LLMs on identifier grounding without retrieval?"
# question = "What is the capital of France?"
# question = "How does open-source model size affect performance?"

questions = [
    "What lipid classes are discussed in this paper?",
    "What metrics did MetaBench use to evaluate LLM performance?",
    "What is the accuracy of LLMs on identifier grounding without retrieval?",
    "What is the capital of France?",
    "How does open-source model size affect performance?"
]

for question in questions:
    response = query_engine.query(question)
    print(f"Q: {question}")
    print(f"A: {response}")