"""
Explainable AI Planner - Interactive CLI
Unified interface for running Explainable A* and AO* heuristic search algorithms.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure local imports work reliably when running directly or from project root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_loader import load_graph
from astar import astar_search
from recommender import recommend_route
from ao_star import ao_star_search

# Standard example AND/OR graph used for AO* demonstrations
SAMPLE_AO_GRAPH = {
    "A": {
        "heuristic": 10,
        "branches": [
            {"type": "OR", "nodes": ["B"], "cost": 1},
            {"type": "AND", "nodes": ["C", "D"], "cost": 2}
        ]
    },
    "B": {
        "heuristic": 6,
        "branches": [
            {"type": "OR", "nodes": ["G1"], "cost": 2}
        ]
    },
    "C": {
        "heuristic": 2,
        "branches": [
            {"type": "OR", "nodes": ["G2"], "cost": 1}
        ]
    },
    "D": {
        "heuristic": 4,
        "branches": [
            {"type": "OR", "nodes": ["G3"], "cost": 1}
        ]
    },
    "G1": {"heuristic": 0, "branches": [], "solved": True},
    "G2": {"heuristic": 0, "branches": [], "solved": True},
    "G3": {"heuristic": 0, "branches": [], "solved": True}
}


def print_solution_tree(tree: Dict[str, Any], indent: int = 0):
    """
    Recursively prints the AND/OR solution tree with clear hierarchy.
    """
    prefix = "  " * indent
    branch_info = f" [{tree.get('branch_type', '')}]" if tree.get("branch_type") else ""
    status = "Solved" if tree.get("solved") else "Unsolved"
    print(f"{prefix}- Node '{tree['node']}'{branch_info} (Cost: {tree['cost']}, Status: {status})")

    for child in tree.get("children", []):
        print_solution_tree(child, indent + 1)


def run_astar_flow():
    """
    Executes the interactive A* Search flow.
    """
    graph_path = PROJECT_ROOT / "data" / "graph.json"

    print("\n--- A* Search Configuration ---")
    try:
        graph = load_graph(graph_path)
    except Exception as err:
        print(f"[Error] Could not load graph data: {err}")
        return

    available_nodes = graph.get("nodes", [])
    print(f"Available nodes in graph: {', '.join(available_nodes)}")

    start_node = input("Enter start node: ").strip()
    goal_node = input("Enter goal node: ").strip()

    try:
        # Run A* Search
        search_result = astar_search(graph, start=start_node, goal=goal_node)
        # Generate Deterministic Recommendation
        recommendation = recommend_route(search_result)

        print("\n========================================")
        print("          A* SEARCH RESULTS")
        print("========================================")
        print(f"Recommended Path : {' -> '.join(recommendation['recommended_path'])}")
        print(f"Total Cost       : {recommendation['cost']}")
        print(f"Number of Steps  : {recommendation['steps']}")
        conf_str = f"{recommendation['confidence']:.2f}" if recommendation['confidence'] is not None else "Not enough evidence"
        print(f"Confidence Score : {conf_str}")
        print(f"Confidence Note  : {recommendation['confidence_explanation']}")
        print(f"\nReason:")
        print(f"{recommendation['recommendation_reason']}")

        print("\n--- Step-by-Step Explainable Audit Trail ---")
        for idx, step in enumerate(search_result.get("audit_trail", []), 1):
            print(f"\nStep {idx}: Selected Node '{step['node']}' (g = {step['g']}, h = {step['h']}, f = {step['f']})")
            print(f"  Reason: {step['reason']}")
            neighbors = step.get("neighbors", [])
            if neighbors:
                print("  Neighboring decisions:")
                for nbr in neighbors:
                    print(f"    - Target '{nbr['node']}' (Tentative Cost: {nbr['tentative_g']}) -> {nbr['action']}")
            else:
                print("  Neighboring decisions: None")

    except ValueError as val_err:
        print(f"\n[Search Error] {val_err}")
    except Exception as err:
        print(f"\n[Unexpected Error] {err}")


def run_ao_star_flow():
    """
    Executes the interactive AO* Search flow.
    """
    print("\n--- AO* Search Configuration ---")
    available_nodes = list(SAMPLE_AO_GRAPH.keys())
    print(f"Available nodes in AND/OR graph: {', '.join(available_nodes)}")

    start_node = input("Enter start node (root): ").strip()

    try:
        result = ao_star_search(SAMPLE_AO_GRAPH, start_node)

        print("\n========================================")
        print("         AO* SEARCH RESULTS")
        print("========================================")
        print(f"Root Node     : {result['root']}")
        print(f"Total Cost    : {result['cost']}")
        print(f"Solved Status : {'Successfully Solved' if result['solved'] else 'Unsolved'}")

        print("\n--- Final Solution Tree ---")
        print_solution_tree(result["solution_tree"])

        print("\n--- Step-by-Step Explainable Audit Trail ---")
        for idx, entry in enumerate(result.get("audit_trail", []), 1):
            chosen = entry.get("chosen_branch")
            branch_desc = (
                f"{chosen.get('type')} branch -> {chosen.get('nodes')} (Edge cost: {chosen.get('cost')})"
                if chosen else "None"
            )
            print(f"\nStep {idx}: Node '{entry['node']}' updated to Cost: {entry['updated_cost']} (Solved: {entry['solved']})")
            print(f"  Reason: {entry['reason']}")
            print(f"  Chosen Branch: {branch_desc}")

    except ValueError as val_err:
        print(f"\n[Search Error] {val_err}")
    except Exception as err:
        print(f"\n[Unexpected Error] {err}")


def main():
    """
    Main interactive menu loop.
    """
    while True:
        print("\n========================================")
        print("       EXPLAINABLE AI PLANNER")
        print("========================================")
        print()
        print("Choose a search algorithm:")
        print("1. A* Search")
        print("2. AO* Search")
        print("3. Exit")
        print()

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            run_astar_flow()
        elif choice == "2":
            run_ao_star_flow()
        elif choice == "3":
            print("\nExiting Explainable AI Planner. Goodbye!\n")
            break
        else:
            print("\n[Invalid Selection] Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
