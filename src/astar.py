import heapq
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory of src to sys.path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_loader import load_graph


def astar_search(graph: Dict[str, Any], start: str, goal: str) -> Dict[str, Any]:
    """
    Executes the Explainable A* search algorithm on a weighted graph with heuristics.

    Args:
        graph: Dictionary containing 'nodes', 'edges', and 'heuristics'.
        start: Starting node identifier.
        goal: Target goal node identifier.

    Returns:
        dict: A dictionary containing:
            - 'path': List of node names representing the optimal path.
            - 'cost': Total numerical cost of the path.
            - 'audit_trail': Step-by-step explainable log of node selections and neighbor evaluations.

    Raises:
        ValueError: If start or goal nodes are invalid, or if no path exists.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", {})
    heuristics = graph.get("heuristics", {})

    # Step 1: Validate that start and goal nodes exist in the graph
    if start not in nodes:
        raise ValueError(f"Invalid start node '{start}'. Node does not exist in graph.")
    if goal not in nodes:
        raise ValueError(f"Invalid goal node '{goal}'. Node does not exist in graph.")

    # Step 2: Initialize the priority queue (frontier)
    # Stored element: (f_score, g_score, current_node, path_taken)
    # heapq pops the element with lowest f_score first (and breaks ties using g_score)
    priority_queue: List[tuple] = []

    start_g = 0
    start_h = heuristics[start]
    start_f = start_g + start_h

    heapq.heappush(priority_queue, (start_f, start_g, start, [start]))

    # Track lowest g_score found so far for each node to evaluate improvements
    lowest_g_score: Dict[str, float] = {start: start_g}

    # Track visited (expanded) nodes so we do not re-expand closed nodes
    visited = set()

    # Step-by-step explainable audit trail
    audit_trail: List[Dict[str, Any]] = []

    while priority_queue:
        # Step 3: Pop the node with the lowest f-score
        f, g, current_node, path = heapq.heappop(priority_queue)

        # Skip if this node has already been expanded
        if current_node in visited:
            continue

        # Mark node as expanded
        visited.add(current_node)
        h = heuristics.get(current_node, 0)

        # Step 4: Check if goal node is reached
        if current_node == goal:
            # Record final audit entry explaining goal was reached
            audit_trail.append({
                "node": current_node,
                "g": g,
                "h": h,
                "f": f,
                "reason": "Goal node reached. Search complete.",
                "neighbors": []
            })
            return {
                "path": path,
                "cost": g,
                "audit_trail": audit_trail
            }

        # Step 5: Evaluate all outgoing neighbors of the current node
        neighbors_evaluation: List[Dict[str, Any]] = []

        for neighbor, edge_weight in edges.get(current_node, {}).items():
            tentative_g = g + edge_weight

            # Case A: Neighbor discovered for the first time
            if neighbor not in lowest_g_score:
                lowest_g_score[neighbor] = tentative_g
                neighbor_h = heuristics.get(neighbor, 0)
                neighbor_f = tentative_g + neighbor_h
                heapq.heappush(priority_queue, (neighbor_f, tentative_g, neighbor, path + [neighbor]))
                
                neighbors_evaluation.append({
                    "node": neighbor,
                    "tentative_g": tentative_g,
                    "action": "added to frontier"
                })

            # Case B: Found a better (lower cost) path to a previously seen neighbor
            elif tentative_g < lowest_g_score[neighbor]:
                lowest_g_score[neighbor] = tentative_g
                neighbor_h = heuristics.get(neighbor, 0)
                neighbor_f = tentative_g + neighbor_h
                heapq.heappush(priority_queue, (neighbor_f, tentative_g, neighbor, path + [neighbor]))

                neighbors_evaluation.append({
                    "node": neighbor,
                    "tentative_g": tentative_g,
                    "action": "updated with a better cost"
                })

            # Case C: Existing path to this neighbor is already better or equal
            else:
                neighbors_evaluation.append({
                    "node": neighbor,
                    "tentative_g": tentative_g,
                    "action": "skipped because existing path was better"
                })

        # Record this node's expansion details and neighbor evaluations in the audit trail
        audit_trail.append({
            "node": current_node,
            "g": g,
            "h": h,
            "f": f,
            "reason": "Selected because it has the lowest f-score.",
            "neighbors": neighbors_evaluation
        })

    # If priority queue empties without reaching goal, no route exists
    raise ValueError(f"No path exists between start node '{start}' and goal node '{goal}'.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    graph_file = project_root / "data" / "graph.json"

    # Load graph using graph_loader
    graph_data = load_graph(graph_file)

    # Execute explainable A* search
    result = astar_search(graph_data, "A", "G")

    print("=== A* Search Results ===")
    print(f"Path: {result['path']}")
    print(f"Total Cost: {result['cost']}")
    print("\n=== Explainable Audit Trail ===")
    for idx, entry in enumerate(result["audit_trail"], 1):
        print(f"\nStep {idx}: Node '{entry['node']}' | g={entry['g']}, h={entry['h']}, f={entry['f']}")
        print(f"  Reason: {entry['reason']}")
        if entry["neighbors"]:
            print("  Neighbors evaluated:")
            for nbr in entry["neighbors"]:
                print(f"    - Node '{nbr['node']}': tentative_g={nbr['tentative_g']} -> {nbr['action']}")
        else:
            print("  Neighbors evaluated: None (Target reached or terminal node)")
