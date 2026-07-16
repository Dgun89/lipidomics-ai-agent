"""
step3_multihop_agent.py

여러 스텝에 걸쳐 도구를 연쇄 호출하는 멀티홉(multi-hop) 에이전트
(프레임워크 없이 직접 루프로 구현 - Option A)

step2와 차이:
    - step2: 도구를 최대 1번만 호출하고 바로 답변
    - step3: 판단 -> 실행 -> 관찰을 반복하여 이전 스텝의 관찰 결과를 다음 판단의 근거로 사용함
            (예: 효소->대사체 목록 조회 후, 각 대사체를 개별적으로 내인성/외인성 분류)
호출형식 (step2와 동일한 스타일 유지)
    'TOOL_CALL: <tool_name> | <argument>' -> 다음 스텝 계속
    'FINAL_ANSWER: <answer>'              -> 루프 종료, 최종 답변
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

MAX_STEPS = 8           # 최대 반복 스텝 횟수 (멈춤 조건 2)
MAX_PARSE_FAILURES = 2  # 연속 파싱 실패 허용 횟수 (멈춤 조건 3)
MAX_REPEAT_FAILURES = 2 # 동일 호출 반복 허용 횟수 (재시도 기회 부여 후 중단)
MAX_ROWS = 15           # 도구 1회 호출당 반환할 최대 행 수 (허브 노드 토큰 폭발 방지)

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


##### 도구(tool) 정의 - step2와 동일 (driver, argument) -> str 시그니처 #####
def get_enzymes_for_metabolite(driver, metabolite_name: str) -> str:
    """대사체 이름 -> 연결된 효소 목록 (최대 MAX_ROWS개까지만 표시)"""
    query = """
    MATCH (m:Metabolite)-[r:INTERACTS_WITH]->(e:Enzyme)
    WHERE toLower(m.name) = toLower($name)
    RETURN e.id AS enzyme, r.source AS source
    """
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query, name=metabolite_name)]
    if not rows:
        return f"No graph data found for metabolite '{metabolite_name}'."
    total = len(rows)
    shown = rows[:MAX_ROWS]
    lines = [f"Metabolite: {metabolite_name}", "Connected enzymes:"]
    lines += [f" - {r['enzyme']} (source: {r['source']})" for r in shown]
    if total > MAX_ROWS:
        lines.append(f" ...and {total - MAX_ROWS} more (this metabolite is a hub with {total} total connections; not all shown)")
    return "\n".join(lines)


def get_metabolites_for_enzyme(driver, enzyme_id: str) -> str:
    """효소 id(EC 번호/유전자 기호) -> 연결된 대사체 목록 (역방향 조회, 최대 MAX_ROWS개까지만 표시)."""
    query = """
    MATCH (m:Metabolite)-[r:INTERACTS_WITH]->(e:Enzyme {id: $eid})
    RETURN m.name AS metabolite, r.source AS source
    """
    with driver.session() as session:
        rows = [dict(r) for r in session.run(query, eid=enzyme_id)]
    if not rows:
        return f"No graph data found for enzyme '{enzyme_id}'."
    total = len(rows)
    shown = rows[:MAX_ROWS]
    lines = [f"Enzyme: {enzyme_id}", "Connected metabolites:"]
    lines += [f" - {r['metabolite']} (source: {r['source']})" for r in shown]
    if total > MAX_ROWS:
        lines.append(f" ...and {total - MAX_ROWS} more (this enzyme is a hub with {total} total connections; not all shown)")
    return "\n".join(lines)


def get_metabolite_info(driver, metabolite_name: str) -> str:
    """대사체 이름 -> 분류/식별자 등 노드 속성."""
    query = """
    MATCH (m:Metabolite)
    WHERE toLower(m.name) = toLower($name)
    RETURN m.classification AS classification, m.inchikey AS inchikey,
           m.smiles AS smiles, m.pubchem AS pubchem, m.kegg AS kegg,
           m.hmdb AS hmdb, m.chebi AS chebi
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


def _format_history(history: list) -> str:
    """지금까지의 (도구, 인자, 관찰결과)를 프롬프트에 넣을 텍스트로 직렬화."""
    if not history:
        return "(아직 없음)"
    lines = []
    for h in history:
        lines.append(
            f"Step {h['step']}: called {h['tool_name']}('{h['argument']}')\n"
            f" -> observation: {h['observation']}"
        )
    return "\n".join(lines)


def _pending_metabolites(history: list) -> list:
    """get_metabolites_for_enzyme에서 언급됐지만 아직 get_metabolite_info로
    확인하지 않은 대사체 이름 목록."""
    mentioned = set()
    checked = set()
    for h in history:
        if h["tool_name"] == "get_metabolites_for_enzyme":
            for line in h["observation"].splitlines():
                m = re.match(r"^\s*-\s*(.+?)\s*\(source:", line)
                if m:
                    mentioned.add(m.group(1).strip())
        if h["tool_name"] == "get_metabolite_info":
            checked.add(h["argument"].strip())
    return sorted(mentioned - checked)


##### 판단(Think) #####
def decide_next_step(question: str, history: list):
    """반환: ("tool_call", tool_name, argument) / ("final_answer", answer, None) / ("parse_error", raw_text, None)."""
    pending = _pending_metabolites(history)
    pending_hint = (
        f"\nIMPORTANT: These metabolites were mentioned but NOT yet checked with "
        f"get_metabolite_info: {', '.join(pending)}. You must call get_metabolite_info "
        f"on each of these (one per step) before you are allowed to give FINAL_ANSWER.\n"
        if pending else ""
    )

    prompt = f"""You are an agent that can call tools multiple times in sequence to answer a question that may require multiple hops of graph lookups.

Available tools:
{_tool_catalog()}

So far you have taken these steps:
{_format_history(history)}
{pending_hint}
Rules:
- Respond with ONLY ONE line, nothing else. No explanation, no reasoning text.
- If you still need more information, respond with EXACTLY:
TOOL_CALL: <tool_name> | <argument>
- The argument must be a plain name/id only (e.g. "NAD+"), never include annotations like "(source: KEGG)".
- If the most recent observation already answers the question, respond with EXACTLY (do not call any tool again):
FINAL_ANSWER: <your complete answer, referencing the specific data observed>
- Never repeat a tool call with the exact same tool name and argument that already appears above.

Question: {question}
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
    except Exception as e:
        return "parse_error", f"(API 호출 실패: {e})", None

    text = resp.choices[0].message.content.strip()

    tool_match = re.search(r"TOOL_CALL:\s*(\w+)\s*\|\s*([^\n]+)", text, re.IGNORECASE)
    final_match = re.search(r"FINAL_ANSWER:\s*(.+)", text, re.IGNORECASE | re.DOTALL)

    if tool_match and (not final_match or tool_match.start() <= final_match.start()):
        tool_name = tool_match.group(1).strip()
        argument = tool_match.group(2).strip().strip("'\"")
        return "tool_call", tool_name, argument

    if final_match:
        answer = re.split(r"\nTOOL_CALL:", final_match.group(1), flags=re.IGNORECASE)[0].strip()
        return "final_answer", answer, None

    return "parse_error", text, None


##### 멈춤 조건 도달 시 지금까지의 관찰만으로 답변 생성 #####
def summarize_history(question: str, history: list, reason: str) -> str:
    prompt = f"""You could not fully complete the investigation ({reason}).
Using ONLY the observations below, give the best partial answer you can and
clearly state what is missing or incomplete.

[Observations so far]
{_format_history(history)}

[Question]
{question}

[Answer]
"""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return (
            f"(요약 생성 실패: {e})\n\n"
            f"[Observations so far]\n{_format_history(history)}"
        )


##### 실행 루프 #####
def run_agent(driver, question: str, max_steps: int = MAX_STEPS):
    print(f"[질문] {question}")
    history = []
    consecutive_parse_failures = 0
    consecutive_repeat_failures = 0

    for step in range(1, max_steps + 1):
        decision, a, b = decide_next_step(question, history)

        if decision == "final_answer":
            print(f"[판단] 정보 충분 -> 최종 답변")
            print(f"[최종 답변] {a}")
            return

        if decision == "parse_error":
            consecutive_parse_failures += 1
            print(f"[판단 실패] 응답 파싱 불가 ({consecutive_parse_failures}회): {a}")
            if consecutive_parse_failures >= MAX_PARSE_FAILURES:
                answer = summarize_history(question, history, "응답 형식을 반복해서 해석하지 못함")
                print(f"[중단 - 파싱 실패 누적] {answer}")
                return
            continue
        consecutive_parse_failures = 0

        tool_name, arg = a, b
        # 모델이 관찰 텍스트를 통째로 인자로 넘기는 실수 방지: "(source: ...)" 꼬리 제거
        arg = re.sub(r"\s*\(source:[^)]*\)\s*$", "", arg, flags=re.IGNORECASE).strip()

        if tool_name not in TOOLS:
            print(f"[판단] 알 수 없는 도구 '{tool_name}' -> 이번 스텝 무시")
            history.append({
                "step": step, "tool_name": tool_name, "argument": arg,
                "observation": f"Error: unknown tool '{tool_name}'.",
            })
            continue

        # 루프 감지: 직전 스텝과 동일한 (도구, 인자) 반복이면 경고 후 재시도 기회 부여
        if history and history[-1]["tool_name"] == tool_name and history[-1]["argument"] == arg:
            consecutive_repeat_failures += 1
            print(f"[경고 - 반복 감지] 동일 호출 반복 ({consecutive_repeat_failures}회): {tool_name}('{arg}')")
            if consecutive_repeat_failures >= MAX_REPEAT_FAILURES:
                answer = summarize_history(question, history, "동일한 도구 호출이 반복됨")
                print(f"[중단 - 반복 누적] {answer}")
                return
            history.append({
                "step": step, "tool_name": tool_name, "argument": arg,
                "observation": "(You already called this exact tool with this exact argument. "
                                "Pick a different tool/argument, or give FINAL_ANSWER using the "
                                "observations already collected.)",
            })
            continue
        consecutive_repeat_failures = 0

        print(f"[판단] 도구 호출 -> {tool_name}('{arg}')")
        func = TOOLS[tool_name][0]
        observation = func(driver, arg)
        print(f"[관찰]\n{observation}")

        history.append({
            "step": step, "tool_name": tool_name, "argument": arg,
            "observation": observation,
        })

    # 최대 스텝 도달 -> 지금까지의 관찰만으로 요약 답변
    print(f"[중단 - 최대 스텝({max_steps}) 도달]")
    answer = summarize_history(question, history, f"최대 스텝 {max_steps}회 도달")
    print(f"[최종 답변(요약)] {answer}")


def main():
    driver = make_driver()
    try:
        tests = [
            # 멀티홉: 효소 -> 대사체 목록 -> 각 대사체 분류
            "Which metabolites are connected to enzyme 1.14.99.2, and is each one endogenous or exogenous?",
            # 싱글홉 회귀 확인용 (step2와 동일하게 동작해야 함)
            "Is Kynurenic acid endogenous or exogenous?",
        ]
        for i, q in enumerate(tests):
            run_agent(driver, q)
            if i < len(tests) - 1:
                print("\n" + "=" * 60 + "\n")
    finally:
        driver.close()


if __name__ == "__main__":
    main()