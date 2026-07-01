# lipidomics-ai-agent

AI 구동형 자율 리피도믹스 프로젝트(WP3·WP4) 준비를 위한 개인 학습 저장소입니다.
가즈사DNA연구소 ABiS Lab 컨소시엄(시마즈·오사카대·가즈사) 담당 업무의 기술 기초를 다지는 용도이며,
박사과정 지원(도호쿠대, 2026년 10월 입학 목표) 연구주제 탐색도 겸합니다.

## 로드맵

| 단계 | 시기 | 내용 | 진행 |
|---|---|---|---|
| 1 | 0~3개월 | RAG 기초 (LlamaIndex/LangChain) + 도메인 기초 (LIPID MAPS, mzTab-M) | ⬜ |
| 2 | 3~6개월 | GraphRAG·지식그래프 (Neo4j, text-to-Cypher) | ⬜ |
| 3 | 6~12개월 | Agentic 통합 (LangGraph, MCP) + QC 이상탐지 ML | ⬜ |
| 4 | 박사 연구 | 다크 리피돔 자율주석(B) + 실시간 온라인 QC 에이전트(D) 심화 | ⬜ |

## 폴더 구조

```
stage1_rag_basics/    1단계: RAG 기초
stage2_graphrag/       2단계: GraphRAG·지식그래프
stage3_agentic/         3단계: Agentic 통합·QC ML
stage4_phd_research/   4단계: 박사 연구 심화
```

## 참고 논문

- MetaBench (Lu et al., arXiv:2510.14944) — LLM 식별자 그라운딩 한계
- QC4Metabolomics (Anal. Chem. 2025) — 실시간 LC-MS QC 공백

## 주의

- 이 저장소는 비공개(private)로 관리합니다. 논문 원문/데이터는 커밋하지 않습니다(`.gitignore` 참조).
