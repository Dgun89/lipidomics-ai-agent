# lipidomics-ai-agent

AI 구동형 자율 리피도믹스 프로젝트(WP3·WP4) 준비를 위한 개인 학습 저장소입니다.
가즈사DNA연구소 ABiS Lab 컨소시엄(시마즈·오사카대·가즈사) 담당 업무의 기술 기초를 다지는 용도이며,
박사과정 지원(도호쿠대, 2026년 10월 입학 목표) 연구주제 탐색도 겸합니다.

## 로드맵

| 단계 | 시기 | 내용 | 진행 |
|---|---|---|---|
| 1 | 0~3개월 | RAG 기초 (LlamaIndex/LangChain) + 도메인 기초 (LIPID MAPS, mzTab-M) | 진행중 |
| 2 | 3~6개월 | GraphRAG·지식그래프 (Neo4j, text-to-Cypher) | 예정 |
| 3 | 6~12개월 | Agentic 통합 (LangGraph, MCP) + QC 이상탐지 ML | 예정 |
| 4 | 박사 연구 | 다크 리피돔 자율주석(B) + 실시간 온라인 QC 에이전트(D) 심화 | 예정 |

## 폴더 구조

```
stage1_rag_basics/
  notebooks/
    step1_load_documents.py     PDF -> Document 객체 (SimpleDirectoryReader)
    step2_build_index.py        Document -> VectorStoreIndex (로컬 임베딩: BAAI/bge-small-en-v1.5)
    step3_query.py              VectorStoreIndex -> 질의응답 (LLM: HuggingFace Inference API, Qwen2.5-7B-Instruct)
    step4_strict_grounding.py   그라운딩 강제 프롬프트 실험 (PromptTemplate)
    step5_balanced_grounding.py 프롬프트 완화 + 청크 재분할 실험 (SentenceSplitter, top_k)
    step6_reranking.py           검색 안정화(재순위화) + 그라운딩 프롬프트 완화 실험
  data/                         PDF 원본 (gitignore 처리, 커밋 안 됨)
stage2_graphrag/
stage3_agentic/
stage4_phd_research/
```

## 1단계 진행 로그

- [x] 환경 설정: llama-index, pypdf, llama-index-readers-file, llama-index-embeddings-huggingface, llama-index-llms-huggingface-api
- [x] step1: PDF 1개(MetaBench 논문) -> 22개 청크(페이지 단위)로 로드 확인
- [x] step2: 로컬 임베딩 모델(HuggingFace BAAI/bge-small-en-v1.5)로 VectorStoreIndex 생성
- [x] step3: 질의응답(query engine) 연결, LLM은 HuggingFace Inference API(Qwen2.5-7B-Instruct) 사용
- [x] 그라운딩 실패 사례 확인: 문서에 없는 질문("프랑스 수도는?")에도 모른다고 답하지 않고 일반 지식으로 답변 -> hallucination 위험 직접 확인
- [x] step4: PromptTemplate으로 "문서에 없으면 모른다고 답하기(그라운딩 강제)" 프롬프트 실험
  - 성공: "프랑스 수도는?" 질문에 "Not found in this document."로 정상 차단
  - 부작용 발견: 지나치게 엄격한 규칙 때문에 문서 안에 실제로 있는 답(지질 클래스 질문)도 거부하는 과잉차단(over-refusal) 현상 확인
- [x] step5: 프롬프트 완화(합리적 근거 허용)로 과잉차단 해소 시도
  - 프롬프트 완화만으로는 해결 안 됨 → 원인이 프롬프트가 아니라 검색(retrieval) 단계에 있음을 발견
  - 청크를 256토큰 단위로 재분할, top_k=5로 확대 → 부분 개선되었으나 완전한 안정화는 아님
  - 결론: 프롬프트 조정 전에 검색 결과(source_nodes)부터 확인하는 진단 순서가 중요함을 확인
- [x] 연구노트 작성: step4_research_note.md, step5_research_note.md (stage4_phd_research/)
- [x] step6: 재순위화(reranking)로 검색 안정화 + 그라운딩 프롬프트 3단계 완화(STRICT→BALANCED→LOOSE) 실험
  - 검색 안정화: SentenceTransformerRerank로 top_k=10 → 재순위화 후 top_n=3, 지질 클래스 질문 검색 성공
  - 프롬프트 완화: 3단계 모두 시도했으나 지질 클래스 질문은 계속 거부됨, 프랑스 수도 질문은 3단계 모두 정상 거부(성공)
  - 결론: 검색 문제와 프롬프트 엄격도 문제는 해결됐으나, "예시 등장 vs 논문 주제로 다룸"을 구분하는 질문 자체의 일반화 요구 수준은 프롬프트만으로 해결 안 됨 — 질문 설계의 문제로 별도 확인 필요
- [ ] 다음: 2단계(GraphRAG) 착수

## 참고 논문

- MetaBench (Lu et al., arXiv:2510.14944) — LLM 식별자 그라운딩 한계
- QC4Metabolomics (Anal. Chem. 2025) — 실시간 LC-MS QC 공백

## 주의

- 이 저장소는 비공개(private)로 관리합니다. 논문 원문/데이터는 커밋하지 않습니다(`.gitignore` 참조).