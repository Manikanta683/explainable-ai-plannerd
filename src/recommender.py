from typing import Any, Dict, List, Optional


def find_best_alternative_cost(
    path: List[str],
    cost: float,
    audit_trail: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Deterministically extracts the best (lowest-cost) alternative complete route cost
    from the A* search audit trail.
    """
    if not audit_trail or not path or len(path) < 2:
        return None

    goal_node = path[-1]
    alt_costs: List[float] = []

    # Map optimal g-scores for nodes in the selected path
    optimal_g_for_node: Dict[str, float] = {}
    for step in audit_trail:
        n = step.get("node")
        g = step.get("g")
        if n in path and g is not None:
            optimal_g_for_node[n] = g

    for entry in audit_trail:
        for nbr in entry.get("neighbors", []):
            nbr_node = nbr.get("node")
            nbr_tentative_g = nbr.get("tentative_g")
            action = nbr.get("action", "")

            # 1. Alternative direct edge reaching the goal
            if nbr_node == goal_node and nbr_tentative_g is not None and nbr_tentative_g > cost:
                alt_costs.append(float(nbr_tentative_g))

            # 2. Alternative route to an intermediate node on the chosen path
            if nbr_node in path and nbr_node != path[0] and nbr_tentative_g is not None:
                opt_g = optimal_g_for_node.get(nbr_node)
                if opt_g is not None and nbr_tentative_g > opt_g:
                    alt_route_cost = cost + (nbr_tentative_g - opt_g)
                    alt_costs.append(float(alt_route_cost))

            # 3. Alternative candidate frontier paths with higher estimated total cost
            if nbr_node not in path and "added" in action:
                entry_f = entry.get("f")
                if entry_f is not None and entry_f > cost:
                    alt_costs.append(float(entry_f))

    valid_alts = [c for c in alt_costs if c > cost]
    if valid_alts:
        return min(valid_alts)

    return None


def recommend_route(search_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produces a deterministic, explainable recommendation and route advantage score
    based on A* search results.

    Args:
        search_result: Dictionary containing 'path', 'cost', and 'audit_trail'.

    Returns:
        dict: Recommendation details including:
            - 'recommended_path': The optimal node sequence.
            - 'cost': Total numerical cost of the path.
            - 'steps': Number of edge transitions in the path.
            - 'recommendation_reason': Clear text explanation of the recommendation.
            - 'confidence': Route advantage score (float between 0 and 1) or None if insufficient evidence.
            - 'confidence_explanation': Detailed text explanation of how confidence was derived.

    Raises:
        ValueError: If search_result is not a dictionary or missing required keys.
    """
    # 1. Validate search_result dictionary
    if not isinstance(search_result, dict):
        raise ValueError("Invalid input: search_result must be a dictionary.")

    # 2. Check for required keys
    required_keys = ["path", "cost", "audit_trail"]
    for key in required_keys:
        if key not in search_result:
            raise ValueError(f"Validation Error: Missing required field '{key}' in search_result.")

    path: List[str] = search_result["path"]
    cost: float = search_result["cost"]
    audit_trail: List[Dict[str, Any]] = search_result["audit_trail"]

    # Validate types of required fields
    if not isinstance(path, list):
        raise ValueError("Validation Error: 'path' must be a list.")
    if not isinstance(cost, (int, float)):
        raise ValueError("Validation Error: 'cost' must be a numeric value.")
    if not isinstance(audit_trail, list):
        raise ValueError("Validation Error: 'audit_trail' must be a list.")

    # 3. Calculate steps (number of edges traversed between nodes)
    steps = max(0, len(path) - 1)
    path_display = " -> ".join(path) if path else "None"

    # 4. Deterministic Route Advantage Score (Confidence)
    alt_cost = find_best_alternative_cost(path, cost, audit_trail)

    if alt_cost is not None and alt_cost > 0:
        # Advantage score formula: (alt_cost - cost) / alt_cost
        advantage_score = max(0.0, min(1.0, (alt_cost - cost) / alt_cost))
        confidence: Optional[float] = round(advantage_score, 4)
        confidence_explanation = (
            f"The selected route costs {cost}, while the best known alternative costs {int(alt_cost) if alt_cost.is_integer() else alt_cost}, "
            f"giving an advantage score of {confidence:.2f}."
        )
        recommendation_reason = (
            f"Route ({path_display}) is recommended with the optimal total cost of {cost} across {steps} step(s). "
            f"{confidence_explanation}"
        )
    else:
        confidence = None
        confidence_explanation = "Confidence: Not enough evidence to determine an alternative route cost."
        recommendation_reason = (
            f"Route ({path_display}) is recommended with total cost {cost} across {steps} step(s). "
            "No valid alternative route could be reliably determined from the search information."
        )

    return {
        "recommended_path": path,
        "cost": cost,
        "steps": steps,
        "recommendation_reason": recommendation_reason,
        "confidence": confidence,
        "confidence_explanation": confidence_explanation
    }


if __name__ == "__main__":
    example_input = {
        "path": ["A", "C", "E", "G"],
        "cost": 7,
        "audit_trail": []
    }

    recommendation = recommend_route(example_input)

    print("=== Route Recommendation ===")
    print(f"Recommended Path       : {recommendation['recommended_path']}")
    print(f"Total Cost             : {recommendation['cost']}")
    print(f"Number of Steps        : {recommendation['steps']}")
    print(f"Confidence Score       : {recommendation['confidence']}")
    print(f"Confidence Explanation : {recommendation['confidence_explanation']}")
    print(f"Recommendation Reason  : {recommendation['recommendation_reason']}")
