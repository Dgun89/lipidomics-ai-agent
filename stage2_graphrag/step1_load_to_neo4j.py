"""
load_to_neo4j.py

metabolites.csv 와 metabolite_enzyme_rels.csv 를 Neo4j Aura에 적재하는 스크립트.

사전 준비:
    pip install neo4j pandas python-dotenv

접속 정보는 코드에 직접 적지 않고, 같은 폴더의 .env 파일에서 읽어옵니다.
.env 파일 예시(.env.example 참고):
    AURA_URI=neo4j+s://xxxx.databases.neo4j.io
    AURA_USER=neo4j
    AURA_PASSWORD=발급받은비밀번호

실행:
    python load_to_neo4j.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ── 1. 접속 정보 (.env에서 읽음) ──────────────────────────────
load_dotenv(override=True)

AURA_URI = os.environ["AURA_URI"]
AURA_USER = os.environ["AURA_USER"]
AURA_PASSWORD = os.environ["AURA_PASSWORD"]

print(f"[디버그] 로드된 AURA_URI: {AURA_URI}")
print(f"[디버그] 로드된 AURA_USER: {AURA_USER}")

# ── 2. CSV 파일 경로 ─────────────────────────────────────────
METABOLITES_CSV = "metabolites.csv"
RELS_CSV = "metabolite_enzyme_rels.csv"

BATCH_SIZE = 500


def load_metabolites(driver, df: pd.DataFrame):
    """Metabolite 노드 생성 (id 기준 MERGE)"""
    query = """
    UNWIND $rows AS row
    MERGE (m:Metabolite {id: row.id})
    SET m.name = row.name,
        m.inchikey = row.inchikey,
        m.smiles = row.smiles,
        m.pubchem = row.pubchem,
        m.kegg = row.kegg,
        m.hmdb = row.hmdb,
        m.chebi = row.chebi,
        m.classification = row.classification
    """
    records = df.where(pd.notnull(df), None).to_dict("records")
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run(query, rows=batch)
            print(f"  Metabolite 적재: {i + len(batch)}/{len(records)}")


def load_relationships(driver, df: pd.DataFrame):
    """Enzyme 노드 생성 + Metabolite-Enzyme 관계 생성"""
    query = """
    UNWIND $rows AS row
    MERGE (e:Enzyme {id: row.enzyme_id})
    WITH e, row
    MATCH (m:Metabolite {id: row.metabolite_id})
    MERGE (m)-[r:INTERACTS_WITH {source: row.source}]->(e)
    """
    records = df.where(pd.notnull(df), None).to_dict("records")
    with driver.session() as session:
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.run(query, rows=batch)
            print(f"  관계 적재: {i + len(batch)}/{len(records)}")


def create_constraints(driver):
    """중복 방지를 위한 유니크 제약조건 생성"""
    with driver.session() as session:
        session.run(
            "CREATE CONSTRAINT metabolite_id IF NOT EXISTS "
            "FOR (m:Metabolite) REQUIRE m.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT enzyme_id IF NOT EXISTS "
            "FOR (e:Enzyme) REQUIRE e.id IS UNIQUE"
        )


def main():
    print("CSV 파일 로딩 중...")
    metabolites_df = pd.read_csv(METABOLITES_CSV)
    rels_df = pd.read_csv(RELS_CSV)
    print(f"  metabolites: {len(metabolites_df)}행")
    print(f"  relationships: {len(rels_df)}행")

    driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

    try:
        print("제약조건 생성 중...")
        create_constraints(driver)

        print("Metabolite 노드 적재 중...")
        load_metabolites(driver, metabolites_df)

        print("관계(Enzyme 포함) 적재 중...")
        load_relationships(driver, rels_df)

        print("완료.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()