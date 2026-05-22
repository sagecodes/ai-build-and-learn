"""
visualize.py

Knowledge graph visualization task.
Queries Cognee's graph, selects the 50 most-connected nodes, and renders
them as a lightweight pyvis HTML that loads instantly in the browser.
"""

import html as _html
from collections import Counter

from config import task_env, configure_cognee

MAX_NODES = 50

# Node type → color
_TYPE_COLORS = {
    "TextSummary": "#6510F4",
    "Entity":      "#A550FF",
    "DocumentChunk": "#0DFF00",
}
_DEFAULT_COLOR = "#747470"


@task_env.task
async def visualize_pipeline() -> str:
    configure_cognee()

    from pyvis.network import Network
    from cognee.infrastructure.databases.graph import get_graph_engine

    engine = await get_graph_engine()
    nodes, edges = await engine.get_graph_data()

    # nodes: list of (id, props_dict)
    # edges: list of (src_id, dst_id, rel_name, props_dict)

    # Pick MAX_NODES nodes with the highest degree
    degree: Counter = Counter()
    for src, dst, *_ in edges:
        degree[src] += 1
        degree[dst] += 1

    all_ids = {n[0] for n in nodes}
    top_ids = {nid for nid, _ in degree.most_common(MAX_NODES)} & all_ids
    if not top_ids:
        top_ids = all_ids

    net = Network(
        height="580px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        cdn_resources="remote",
    )
    net.barnes_hut(spring_length=120, spring_strength=0.04, damping=0.09)

    for nid, props in nodes:
        if nid not in top_ids:
            continue
        label = (props.get("name") or props.get("type") or str(nid)[:20]).strip() or str(nid)[:20]
        ntype = props.get("type", "")
        color = _TYPE_COLORS.get(ntype, _DEFAULT_COLOR)
        tooltip = f"{ntype}\n{label}"
        net.add_node(str(nid), label=label[:30], title=tooltip, color=color, size=12)

    added = {str(nid) for nid, _ in nodes if nid in top_ids}
    for src, dst, rel, *_ in edges:
        if str(src) in added and str(dst) in added:
            net.add_edge(str(src), str(dst), title=rel, label="", arrows="to")

    graph_html = net.generate_html()
    # gr.HTML injects via innerHTML which blocks <script> execution.
    # Wrapping in a srcdoc iframe gives it a proper browsing context.
    srcdoc = _html.escape(graph_html, quote=True)
    return f'<iframe srcdoc="{srcdoc}" style="width:100%;height:600px;border:none;background:#1a1a2e;"></iframe>'
