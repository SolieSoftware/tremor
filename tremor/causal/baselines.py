import json
from pathlib import Path
from typing import Optional

# Module-level baselines, loaded once at startup
_baselines: dict = {}


def load_baselines(path: str) -> None:
    """Load IRF/FEVD baselines from a JSON file.

    Expected structure:
    {
        "source_node": {
            "target_node": {
                "direction": "positive" | "negative",
                "responses": [0.0, 0.1, 0.05, ...]  // IRF values by lag (index = lag in weeks)
            }
        }
    }
    """
    global _baselines
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baselines file not found: {path}")
    with open(p) as f:
        _baselines = json.load(f)


def get_expected_response(source_node: str, target_node: str, lag: int) -> Optional[float]:
    """Get the expected IRF response magnitude for a source→target at a given lag."""
    source_data = _baselines.get(source_node, {})
    target_data = source_data.get(target_node, {})
    responses = target_data.get("responses", [])
    if lag < len(responses):
        return responses[lag]
    return None


def get_expected_direction(source_node: str, target_node: str) -> Optional[str]:
    """Get the expected direction of response: 'positive' or 'negative'."""
    source_data = _baselines.get(source_node, {})
    target_data = source_data.get(target_node, {})
    return target_data.get("direction")
