import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    token=os.getenv("HF_TOKEN"),
)
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)

# step5 balanced 버전 그대로 사용 (검색은 이미 reranking으로 안정화됨)
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

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": balanced_prompt})

# 일반화 요구 있음 vs. 없음 비교
questions = [
    # 일반화 요구: 논문이 다룸
    "What lipid classes are discussed in this paper?",
    # 일반화 없음: 예시로 언급된 것은?
    "What lipid class is given as an example in the taxonomy classification question in this paper?",
    # 여전히 거부되어야 함
    "What is the capital of France?"
]

for q in questions:
    response = query_engine.query(q)
    print(f"Q: {q}")
    print(f"A: {response}\n")
    