import csv
from pathlib import Path
from typing import Optional

import networkx as nx

# Module-level graph instance, loaded once at startup
causal_network: nx.DiGraph = nx.DiGraph()


def load_network(path: str) -> None:
    """Load the causal network from a GraphML file or Granger results CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Network file not found: {path}")

    if p.suffix == ".graphml":
        loaded = nx.read_graphml(p)
    elif p.suffix == ".csv":
        loaded = _load_from_granger_csv(p)
    else:
        raise ValueError(f"Unsupported network file format: {p.suffix}")

    causal_network.clear()
    causal_network.update(loaded)


def _load_from_granger_csv(path: Path) -> nx.DiGraph:
    """Build a directed graph from Granger causality results CSV.

    Expected columns: cause, effect, f_statistic, p_value, lag
    """
    g = nx.DiGraph()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g.add_edge(
                row["cause"],
                row["effect"],
                f_statistic=float(row.get("f_statistic", 0)),
                p_value=float(row.get("p_value", 1)),
                lag=int(row.get("lag", 1)),
            )
    return g


def get_downstream_nodes(node: str) -> list[str]:
    """Get all nodes that this node has direct edges TO."""
    if node not in causal_network:
        return []
    return list(causal_network.successors(node))


def get_upstream_nodes(node: str) -> list[str]:
    """Get all nodes that have edges TO this node."""
    if node not in causal_network:
        return []
    return list(causal_network.predecessors(node))


def get_edge_info(source: str, target: str) -> Optional[dict]:
    """Return edge metadata (f_statistic, lag, p_value) for an edge."""
    if not causal_network.has_edge(source, target):
        return None
    return dict(causal_network.edges[source, target])


def get_transmission_path(source: str, target: str) -> Optional[list[str]]:
    """Find the shortest directed path between two nodes."""
    try:
        return nx.shortest_path(causal_network, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def get_all_edges() -> list[dict]:
    """Return all edges with metadata."""
    edges = []
    for source, target, data in causal_network.edges(data=True):
        edges.append({"source": source, "target": target, **data})
    return edges
