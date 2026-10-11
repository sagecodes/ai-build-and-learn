"""
ingest/enrichment.py — entity resolution, community detection, and summarization tasks

resolve_entities:      merge near-duplicate Entity nodes by embedding similarity.
detect_communities:    run Louvain community detection over the entity graph.
summarize_communities: generate Claude summaries for each community.
"""

import json
from collections import defaultdict
from itertools import combinations

import networkx as nx
import community as louvain_community  # python-louvain
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CLAUDE_MODEL,
    EMBED_MODEL,
    ENTITY_MERGE_THRESHOLD,
    LOUVAIN_RESOLUTION,
    anthropic_client,
    neo4j_driver,
    task_env,
)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@task_env.task
async def resolve_entities() -> str:
    """
    Merge near-duplicate Entity nodes in Neo4j using embedding cosine similarity.

    Returns:
        JSON summary — {merges_performed}.
    """
    driver = neo4j_driver()
    model = SentenceTransformer(EMBED_MODEL)

    with driver:
        with driver.session() as session:
            rows = session.run("MATCH (e:Entity) RETURN e.name AS name").data()

    names = [r["name"] for r in rows]
    if len(names) < 2:
        return json.dumps({"merges_performed": 0})

    embeddings = model.encode(names)
    name_to_vec = {name: embeddings[i] for i, name in enumerate(names)}

    merge_pairs = []
    for a, b in combinations(names, 2):
        if _cosine_sim(name_to_vec[a], name_to_vec[b]) >= ENTITY_MERGE_THRESHOLD:
            merge_pairs.append((a, b))

    merges_performed = 0
    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for keep_name, drop_name in merge_pairs:
                session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(drop:Entity {name: $drop})
                    MATCH (keep:Entity {name: $keep})
                    MERGE (c)-[:MENTIONS]->(keep)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                session.run(
                    """
                    MATCH (drop:Entity {name: $drop})-[r:RELATED]->(other:Entity)
                    MATCH (keep:Entity {name: $keep})
                    MERGE (keep)-[:RELATED {type: r.type, description: r.description}]->(other)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                session.run(
                    """
                    MATCH (other:Entity)-[r:RELATED]->(drop:Entity {name: $drop})
                    MATCH (keep:Entity {name: $keep})
                    MERGE (other)-[:RELATED {type: r.type, description: r.description}]->(keep)
                    """,
                    drop=drop_name,
                    keep=keep_name,
                )
                session.run(
                    "MATCH (e:Entity {name: $drop}) DETACH DELETE e",
                    drop=drop_name,
                )
                merges_performed += 1

    return json.dumps({"merges_performed": merges_performed})


@task_env.task
async def detect_communities() -> str:
    """
    Run Louvain community detection over the entity graph and write community
    IDs back to Neo4j Entity nodes.

    Returns:
        JSON summary — {communities_found, entities_assigned}.
    """
    driver = neo4j_driver()

    with driver:
        with driver.session() as session:
            entity_rows = session.run("MATCH (e:Entity) RETURN e.name AS name").data()
            rel_rows = session.run(
                "MATCH (a:Entity)-[:RELATED]->(b:Entity) "
                "RETURN a.name AS source, b.name AS target"
            ).data()

    G = nx.Graph()
    G.add_nodes_from(r["name"] for r in entity_rows)
    G.add_edges_from((r["source"], r["target"]) for r in rel_rows)

    partition = louvain_community.best_partition(G, resolution=LOUVAIN_RESOLUTION)

    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for entity_name, community_id in partition.items():
                session.run(
                    "MATCH (e:Entity {name: $name}) SET e.community_id = $cid",
                    name=entity_name,
                    cid=community_id,
                )

    communities_found = len(set(partition.values()))
    return json.dumps({
        "communities_found": communities_found,
        "entities_assigned": len(partition),
    })


@task_env.task
async def summarize_communities() -> str:
    """
    Generate a natural-language summary for each community and store it in Neo4j.

    Returns:
        JSON summary — {communities_summarized}.
    """
    driver = neo4j_driver()

    with driver:
        with driver.session() as session:
            rows = session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN e.name AS name, e.type AS type, "
                "       e.description AS description, e.community_id AS cid"
            ).data()
            rel_rows = session.run(
                "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
                "WHERE a.community_id IS NOT NULL "
                "RETURN a.name AS source, b.name AS target, "
                "       r.type AS rel_type, r.description AS description, "
                "       a.community_id AS cid"
            ).data()

    community_entities: dict = defaultdict(list)
    community_rels: dict = defaultdict(list)

    for r in rows:
        community_entities[r["cid"]].append(r)
    for r in rel_rows:
        community_rels[r["cid"]].append(r)

    client = anthropic_client()
    communities_summarized = 0

    driver2 = neo4j_driver()
    with driver2:
        with driver2.session() as session:
            for cid, entities in community_entities.items():
                rels = community_rels.get(cid, [])

                entity_lines = "\n".join(
                    f"- {e['name']} ({e['type']}): {e['description']}"
                    for e in entities
                )
                rel_lines = "\n".join(
                    f"- {r['source']} --[{r['rel_type']}]--> {r['target']}: {r['description']}"
                    for r in rels
                ) or "None"

                prompt = (
                    "You are summarizing a cluster of related concepts from Everstorm Outfitters "
                    "policy and product documents. Write a concise 2-3 sentence summary that "
                    "describes what this group of entities is about and how they relate to each other.\n\n"
                    f"Entities:\n{entity_lines}\n\n"
                    f"Relationships:\n{rel_lines}"
                )

                response = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary_text = response.content[0].text.strip()

                session.run(
                    """
                    MERGE (comm:Community {id: $cid})
                    SET comm.summary = $summary
                    """,
                    cid=cid,
                    summary=summary_text,
                )
                for ent in entities:
                    session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (comm:Community {id: $cid})
                        MERGE (e)-[:BELONGS_TO]->(comm)
                        """,
                        name=ent["name"],
                        cid=cid,
                    )
                communities_summarized += 1

    return json.dumps({"communities_summarized": communities_summarized})
