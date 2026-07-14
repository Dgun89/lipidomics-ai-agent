"""
step2_multi_tool_agetn.py

여러 도구 중 LLM이 스스로 골라 호출하는 에이전트 (프레임워크 없이 직접 구현)
step1과 차이:
    - step1: 도구 1개, "쓸까/말까"만 판단함
    - step2: 도구 n개, "어떤 도구를 어떤 인자로 쓸까"까지 판단
호출 형식: 'TOOL_CALL: <tool_name> | <argument>' 아니면 직접 답변
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

load_dotenv(override=True)

AURA_URI = os.environ["AURA_URI"]
AURA_USER = os.environ["AURA_USER"]
AURA_PASSWORD = os.environ["AURA_PASSWORD"]
GROQ_MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def make_driver():
    driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, AuthError) as e:
        raise SystemExit(
            f"[연결 실패] Neo4j Aura에 접속할 수 없습니다: {e}\n"
            f" - console.neo4j.io에서 인스턴스 상태 확인\n"
            f" - stage3_agentic/.env의 AURA_URI / AURA_USER / AURA_PASSWORD 갱신"
        )
    return driver

##### 도구(tool) 정의 - 모두 (driver, argument) -> str 시그니처 #####
def get_enzymes_for_metabolite(driver, metabolite_name: str) -> str:
    """대사체 이름 -> 연결된 효소 목록"""
    query = """
    MATCH (m:Metabolite)-[r:INTERACTS_WITH]->(e:Enzyme)
    WHERE toLower(m.name) = toLower($name)
    RETURN e.id AS enzyme, r.source AS source
"""
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query, name=metabolite_name)]
    if not rows:
        return f"No graph data found for metabolite '{metabolite_name}'."
    lines = [f"Metabolite: {metabolite_name}", "Connected enzymes:"]
    lines += [f" - {r['enzyme']} (source: {r['source']})" for r in rows]
    return "\n".join(lines)

def get_metabolites_for_enzyme(driver, enzyme_id: str) -> str:
    """효소 id(EC 번호/유전자 기호) -> 연결된 대사체 목록 (역방향 조회)."""
    query = """
    MATCH (m:Metabolite)-[r:INTERACTS_WITH]->(e:Enzyme {id: $eid})
    RETURN m.name AS metabolite, r.source AS source
    """
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query, eid=enzyme_id)]
    if not rows:
        return f"No graph data found for enzyme '{enzyme_id}'."
    lines = [f"Enzyme: {enzyme_id}", "Connected metabolites:"]
    lines += [f" - {r['metabolite']} (source: {r['source']})" for r in rows]
    return "\n".join(lines)

def get_metabolite_info(driver, metabolite_name: str) -> str:
    """대사체 이름 -> 분류/식별자 등 노드 속성."""
    query = """
    MATCH (m:Metabolite)
    WHERE toLower(m.name) = toLower($name)
    RETURN m.classification AS classification, m.inchikey AS inchikey,
           m.smiles AS smiles, m.pubchem AS pubchem, m.kegg AS kegg,
           m.hmdb AS hmdb, m.chebi as chebi
    """
    with driver.session() as session:
        rec = session.run(query, name=metabolite_name).single()
    if rec is None:
        return f"No graph data found for metabolite '{metabolite_name}'."
    d = dict(rec)
    lines = [f"Metabolite: {metabolite_name}"]
    lines += [f" - {k}: {v}" for k, v in d.items() if v not in (None, "")]
    return "\n".join(lines)

##### 도구 레지스트리: 이름 -> (함수, 설명) #####
TOOLS = {
    "get_enzymes_for_metabolite": (
        get_enzymes_for_metabolite,
        "argument = metabolite name. Return enzymes connected to that metabolite.", 
    ), 
    "get_metabolites_for_enzyme": (
        get_metabolites_for_enzyme,
        "argument = enzyme id (EC number or gene symbol). Return metabolites connected to that enzyme.",
    ),
    "get_metabolite_info": (
    get_metabolite_info,
    "argument = metabolite name. Return classification and identifiers (InChIKey, SMILES, PubChem, KEGG, HMDB, ChEBI).",
    ),
}

def _tool_catalog() -> str:
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())

def decide_tool_call(question: str):
    """반환: (tool_name, argument) 또는 (None, direct_answer)."""
    prompt = f"""You are an agent with access to these tools:
{_tool_catalog()}

Decide whether answering the question needs one of these tools.
- If it does, respond with EXACTLY this format and nothing else:
TOOL_CALL: <tool_name> | <argument>
- If it does not, answer the question directly in plain text.

Question: {question}
"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}], max_tokens=200
    )
    text = resp.choices[0].message.content.strip()

    m = re.match(r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text

def answer_with_context(question: str, context: str) -> str:
    prompt = f"""The following is information retrieved from a graph database.
    If the answer is not contained in this information, respond only with "Not found in the provided data."
    Do not guess or use general knowledge.

    [Graph Data]
    {context}

    [Question]
    {question}

    [Answer]
    """

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}], max_tokens=300
    )
    return resp.choices[0].message.content

##### 실행 루프 #####
def run_agent(driver, question: str):
    print(f"[질문] {question}")
    tool_name, arg = decide_tool_call(question)

    if tool_name is None:
        print("[판단] 도구 호출 불필요 -> 직접 답변")
        print(f"[답변] {arg}")
        return
    
    if tool_name not in TOOLS:
        print(f"[판단] 알 수 없는 도구 '{tool_name}' -> 직접 답변으로 대체")
        print(f"[답변] Not found in the provided data.")
        return 
    
    print(f"[판단] 도구 호출 -> {tool_name}('{arg}')")
    func = TOOLS[tool_name][0]
    context = func(driver, arg)
    print(f"[도구 실행 결과]\n{context}")

    final = answer_with_context(question, context)
    print(f"[최종 답변] {final}")

def main():
    driver = make_driver()
    try:
        tests = [
            "What enzymes are associated with kynurenic acid?", # tool 1
            "Which metabolites are connected to enzyme 1.14.99.2?", # tool 2
            "Is Kynurenic acid endogenous or exogenous?", # tool 3
            "What is the capital of France?", # 직접 답변
        ]
        for i, q in enumerate(tests):
            run_agent(driver, q)
            if i < len(tests) - 1:
                print("\n" + "=" * 60 + "\n")

    finally:
        driver.close()

if __name__ == "__main__":
    main()