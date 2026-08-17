import json
from pathlib import Path
from typing import Any, Dict, Union


def load_graph(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and validates a graph configuration from a JSON file.

    Args:
        path: Filepath to the graph JSON file.

    Returns:
        dict: The loaded and validated graph data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON validation fails or required components are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Graph file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in graph file: {e}") from e

    if not isinstance(graph, dict):
        raise ValueError("Graph data must be a JSON object (dictionary).")

    # 3. Validate required keys
    required_keys = ["nodes", "edges", "heuristics"]
    for key in required_keys:
        if key not in graph:
            raise ValueError(f"Validation Error: Missing required key '{key}'.")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    heuristics = graph.get("heuristics")

    if not isinstance(nodes, list):
        raise ValueError("Validation Error: 'nodes' must be a list.")
    if not isinstance(edges, dict):
        raise ValueError("Validation Error: 'edges' must be a dictionary.")
    if not isinstance(heuristics, dict):
        raise ValueError("Validation Error: 'heuristics' must be a dictionary.")

    # 4. Validate every node has an entry in 'edges'
    for node in nodes:
        if node not in edges:
            raise ValueError(f"Validation Error: Node '{node}' is missing an entry in 'edges'.")

    # 5. Validate every node has a heuristic value
    for node in nodes:
        if node not in heuristics:
            raise ValueError(f"Validation Error: Node '{node}' is missing a heuristic value.")

    return graph


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    graph_path = project_root / "data" / "graph.json"

    try:
        loaded_graph = load_graph(graph_path)
        print("Graph loaded successfully.")
        print(f"Nodes: {loaded_graph['nodes']}")
        print(f"Edges: {loaded_graph['edges']}")
    except Exception as err:
        print(f"Error loading graph: {err}")
