import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
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

# 문서에 없으면 모른다고 답하도록 강제하는 프롬프트
strict_prompt = PromptTemplate(
    "You must answer ONLY using the context below. "
    "If the answer is not explicitly contained in the context, "
    "respond exactly with: 'Not found in this document.' \n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

query_engine = index.as_query_engine()
query_engine.update_prompts({"response_synthesizer:text_qa_template": strict_prompt})

questions = [
    "What lipid classes are discussed in this paper?",
    "What metrics did MetaBench use to evaluate LLM performance?",
    "What is the accuracy of LLMs on identifier grounding without retrieval?",
    "What is the capital of France?",
    "How does open-source model size affect performance?"
]

for q in questions:
    response = query_engine.query(q)
    print(f"Q: {q}")
    print(f"A: {response}\n")
