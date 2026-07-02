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
    step1_load_documents.py   PDF -> Document 객체 (SimpleDirectoryReader)
    step2_build_index.py      Document -> VectorStoreIndex (로컬 임베딩: BAAI/bge-small-en-v1.5)
    step3_query.py            VectorStoreIndex -> 질의응답 (LLM: HuggingFace Inference API, Qwen2.5-7B-Instruct)
    step4_strict_grounding.py 그라운딩 강제 프롬프트 실험 (PromptTemplate)
  data/                       PDF 원본 (gitignore 처리, 커밋 안 됨)
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
- [ ] 다음: 과잉차단 완화 — "명시적으로 없으면" 대신 "충분한 근거가 없으면" 등으로 프롬프트 균형점 조정

## 참고 논문

- MetaBench (Lu et al., arXiv:2510.14944) — LLM 식별자 그라운딩 한계
- QC4Metabolomics (Anal. Chem. 2025) — 실시간 LC-MS QC 공백

## 주의

- 이 저장소는 비공개(private)로 관리합니다. 논문 원문/데이터는 커밋하지 않습니다(`.gitignore` 참조).