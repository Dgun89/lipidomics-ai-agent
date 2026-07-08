"""
step1_simple_agent.py

도구(tool) 1개를 사용하는 가장 단순한 형태의 에이전트.
LangChain 같은 프레임워크 없이, 프롬프트 설계와 문자열 파싱만으로
"LLM이 스스로 도구 호출 여부를 판단"하는 구조를 직접 구현함

핵심 개념 (ReAct 패턴의 단순화 버전):
    질문 -> LLM 판단 (도구 필요? Y/N) -> (필요시) 도구 실행 -> 결과를 LLM에 재전달 -> 최종 답변

stage2(GraphRAG)와의 차이:
    - stage2: 항상 정해진 순서대로 실행(무조건 조회 -> 무조건 LLM에 전달)
    - stage3: LLM이 "이 질문은 조회가 필요한가?"를 스스로 먼저 판단

사전 준비: pip install huggingface_hub python-dotenv

실행: python step1_simple_agent.py
"""

import os
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from neo4j import GraphDatabase

load_dotenv(override=True)

AURA_URI = os.environ["AURA_URI"]
AURA_USER = os.environ["AURA_USER"]
AURA_PASSWORD = os.environ["AURA_PASSWORD"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

### 도구(tool) 정의 ###
def get_enzymes_for_metabolite(driver, metabolite_name: str) -> str:
    """Neo4j에서 특정 대사체와 연결된 효소 목록을 조회하는 도구."""
    query = """
    MATCH (m:Metabolite {name: $name})-[r:INTERACTS_WITH]->(e:Enzyme)
    RETURN e.id AS enzyme, r.source AS source
    """
    with driver.session() as session:
        result = session.run(query, name=metabolite_name)
        rows = [dict(record) for record in result]

    if not rows:
        return f"No graph data found for '{metabolite_name}"
    
    lines = [f"Metabolite: {metabolite_name}", "Connected enzymes:"]
    for row in rows:
        lines.append(f" - {row['enzyme']} (source: {row['source']})")
    return "\n".join(lines)

### 에이전트 판단 단계 ###
def decide_tool_call(question: str) -> str | None:
    """
    LLM에게 도구 호출 필요 여부를 판단시킨다.
    필요하면, 'TOOL_CALL: <metabolite_name>' 형식으로 답하도록 지시하고,
    그 형식이면 대사체 이름을 추출해서 반환, 아니면 None 반환
    """
    prompt = f"""You have access to one tool:
    
    get_enzyme_for_metabolite(metabolite_name): returns the list of enzymes connected to a given metabolite in a graph database.

    Decide whether answering the following question requires calling this tool.
    - If it does, respond with EXACTLY this format and nothing else:
    TOOL_CALL: <metabolite_name>
    - If it does not require the tool, answer the question directly in plain text.

    Question: {question}
    """

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    text = response.choices[0].message.content.strip()

    match = re.match(r"TOOL_CALL:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None, text # 도구 불필요: (None, 직접 답변)

def answer_with_context(question: str, context: str) -> str:
    """도구 실행 결과(컨텍스트)를 근거로 최종 답변을 생성한다."""
    prompt = f"""The following is information retrieved from a graph database. 
    If the answer is not contained in this information, respond only with "Not found in the provided data."
    Do not guess or use general knowledge.NotADirectoryError
    
    [Graph Data]
    {context}

    [Question]
    {question}

    [Answer]
    """
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content

### 에이전트 실행 루프 ###
def run_agent(driver, question: str):
    print(f"[질문] {question}")

    decision = decide_tool_call(question)

    if isinstance(decision, tuple):
        #도구 불필요 -> 직접 답변
        _, direct_answer = decision
        print("[판단] 도구 호출 불필요 -> 직접 답변")
        print(f"[답변] {direct_answer}")
        return
    
    metabolite_name = decision
    print(f"[판단] 도구 호출 필요 -> get_enzymes_for_metabolite('{metabolite_name}')")

    context = get_enzymes_for_metabolite(driver, metabolite_name)
    print(f"[도구 실행 결과]\n{context}")

    final_answer = answer_with_context(question, context)
    print(f"[최종 답변] {final_answer}")

def main():
    driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
    try:
        # 도구가 필요한 질문
        run_agent(driver, "what enzymes are associated with Kynurenic acid?")
        print("\n" + "=" * 60 + "\n")
        # 도구가 필요 없는 질문 (일반 상식)
        run_agent(driver, "what is the capital of France?")
    finally:
        driver.close()

if __name__ == "__main__":
    main()