"""
Explainable AI Planner
Main entry point for executing graph loading, A* search, and deterministic recommendation.
"""

import sys
from pathlib import Path

from src.graph_loader import load_graph
from src.astar import astar_search
from src.recommender import recommend_route


def print_report(start_node: str, goal_node: str, search_result: dict, recommendation: dict):
    """
    Prints a clean, formatted report explaining the planning and recommendation results.
    """
    print("=== Explainable AI Planner ===")
    print()
    print(f"Start: {start_node}")
    print(f"Goal: {goal_node}")
    print()
    print("--- Recommended Route ---")
    
    path_formatted = " -> ".join(recommendation["recommended_path"])
    print(f"Path: {path_formatted}")
    print(f"Total Cost: {recommendation['cost']}")
    print(f"Steps: {recommendation['steps']}")
    print(f"Confidence: {recommendation['confidence']}")
    print()
    print("Reason:")
    print(recommendation["recommendation_reason"])
    print()
    print("--- A* Audit Trail ---")
    print()

    for idx, entry in enumerate(search_result.get("audit_trail", []), 1):
        node = entry.get("node")
        g = entry.get("g")
        h = entry.get("h")
        f = entry.get("f")
        reason = entry.get("reason")
        neighbors = entry.get("neighbors", [])

        print(f"Step {idx}: Selected Node '{node}' (g = {g}, h = {h}, f = {f})")
        print(f"  Reason: {reason}")

        if neighbors:
            print("  Neighboring decisions:")
            for nbr in neighbors:
                nbr_node = nbr.get("node")
                tentative_g = nbr.get("tentative_g")
                action = nbr.get("action")
                print(f"    - Target Node '{nbr_node}' (tentative cost: {tentative_g}) -> {action}")
        else:
            print("  Neighboring decisions: None (Goal reached or terminal node)")
        print()


def main():
    # Define paths relative to this script
    project_root = Path(__file__).resolve().parent
    graph_path = project_root / "data" / "graph.json"

    start_node = "A"
    goal_node = "G"

    try:
        # 1. Load the graph
        graph = load_graph(graph_path)

        # 2. Run A* search
        search_result = astar_search(graph, start=start_node, goal=goal_node)

        # 3. Generate deterministic recommendation
        recommendation = recommend_route(search_result)

        # 4. Print clean, readable report
        print_report(start_node, goal_node, search_result, recommendation)

    except FileNotFoundError as fnf_err:
        print(f"\n[Error] Unable to locate graph file: {fnf_err}")
        sys.exit(1)
    except ValueError as val_err:
        print(f"\n[Validation Error] {val_err}")
        sys.exit(1)
    except Exception as err:
        print(f"\n[Unexpected Error] An unexpected error occurred: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
