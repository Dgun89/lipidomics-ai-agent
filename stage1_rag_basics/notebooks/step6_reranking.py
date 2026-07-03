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

# 검색 안정화: 1차 top_k=10 -> 재순위화 상위 3개 재선정
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
)

# 그라운딩 강제: 합리적 근거가 있으면 허용, 전혀 없으면 거부 (step5 balanced 버전)
# balanced_prompt = PromptTemplate(
#     "Answer using the context below. You may summarize or infer an answer "
#     "if the context provides reasonable supporting evidence, even if not phrased "
#     "identically to the question. "
#     "Only respond with 'Not found in this document.' if the context has "
#     "no reasonable evidence at all related to the question.\n\n"
#     "Context:\n{context_str}\n\n"
#     "Question: {query_str}\n"
#     "Answer:"
# )

# 그라운딩 완화 2차: "근거 있음"의 기준을 예시 및 부분 언급까지 확장 (프랑스 수도는 여전히 거부되어야 정상임)
loose_prompt = PromptTemplate(
    "Answer the question using the context below. "
    "Even a single example, mention, or partial reference in the context is enough evidence to answer. "
    "Only respond with 'Not found in this document.' if the context has "
    "absolutely no relation to the question topic.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n"
    "Answer:"
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
query_engine.update_prompts({"response_synthesizer:text_qa_template": loose_prompt})

questions = [
    "What lipid classes are discussed in this paper?",
    "What is the capital of France?",
    "What is the accuracy of LLMs on identifier grounding without retrieval?",
]

for q in questions:
    response = query_engine.query(q)
    print(f"Q: {q}")
    print(f"A: {response}\n")