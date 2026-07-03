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

from llama_index.core.node_parser import SentenceSplitter

Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

# step4: 너무 엄격한 버전(비교용, 그대로 둠)
strict_prompt = PromptTemplate(
    "You must answer ONLY using the context below. "
    "If the answer is not explicitly contained in the context, "
    "respond exactly with: 'Not found in this document.'\n\n"
    "Context: \n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

# step5: 완화된 버전 - "명시적으로" 대신 "충분한 근거"로 기준 변경
balanced_prompt = PromptTemplate(
    "Answer using the context below. You may summarize or infer an answer "
    "if the context provides reasonable supporting evidence, even if not phrased "
    "identically to the question. "
    "Only respond with 'Not found in this document.' if the context has "
    "no reasonable evidence at all related to the question.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

questions = [
    # "What lipid classes are discussed in this paper?",
    # "what is the capital of France?",
    "what is an example benchmark question used to test lipid taxonomy classification?",
    "what is the capital of France?",
]

for label, prompt in [("STRICT (step4)", strict_prompt), ("BALANCED (step5)", balanced_prompt)]:
    print(f"\n==== {label} ====")
    query_engine = index.as_query_engine(similarity_top_k=5)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt})

    for q in questions:
        response = query_engine.query(q)
        print(f"Q: {q}")
        print(f"A: {response}\n")

        for node in response.source_nodes:
            print(node.text[:200])
            print("---")
        
