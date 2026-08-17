"""
AO* (AND/OR Graph Search) Algorithm Implementation
Provides explainable heuristic search for AND/OR graphs with subproblem decomposition.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


class AOStarSearch:
    def __init__(self, graph: Dict[str, Any]):
        """
        Initializes the AO* solver with an AND/OR graph structure.

        Graph format:
        {
            "node_name": {
                "heuristic": int/float,
                "branches": [
                    {"type": "OR", "nodes": ["B"], "cost": 1},
                    {"type": "AND", "nodes": ["C", "D"], "cost": 2}
                ],
                "solved": bool (optional, defaults to False for non-terminals)
            }
        }
        """
        self.graph = graph
        self.heuristics: Dict[str, float] = {}
        self.solved_status: Dict[str, bool] = {}
        self.best_branch: Dict[str, Optional[Dict[str, Any]]] = {}
        self.parents: Dict[str, Set[str]] = {}
        self.audit_trail: List[Dict[str, Any]] = []

        # Initialize node heuristics and status
        for node, data in self.graph.items():
            self.heuristics[node] = float(data.get("heuristic", 0))
            self.solved_status[node] = bool(data.get("solved", False))
            self.best_branch[node] = None
            if node not in self.parents:
                self.parents[node] = set()

        # Build parent references for bottom-up cost propagation
        for parent_node, data in self.graph.items():
            for branch in data.get("branches", []):
                for child in branch.get("nodes", []):
                    if child not in self.parents:
                        self.parents[child] = set()
                    self.parents[child].add(parent_node)

    def calculate_branch_cost(self, branch: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Computes the cost of a given branch (AND / OR) and checks if all child nodes are solved.

        For OR (single child): cost = edge_cost + h(child)
        For AND (multiple children): cost = edge_cost + sum(h(child_i))
        """
        edge_cost = branch.get("cost", 1.0)
        children = branch.get("nodes", [])
        
        # Calculate sum of child heuristics
        child_heuristics_sum = sum(self.heuristics.get(child, 0) for child in children)
        total_branch_cost = edge_cost + child_heuristics_sum

        # Branch is solved only if ALL constituent child nodes are solved
        all_children_solved = len(children) > 0 and all(self.solved_status.get(c, False) for c in children)

        return total_branch_cost, all_children_solved

    def update_node_cost(self, node: str) -> bool:
        """
        Recalculates the minimum cost for a node among its branches (OR/AND),
        updates its best branch pointer, and checks if the node is solved.

        Returns True if the cost or solved status changed.
        """
        branches = self.graph.get(node, {}).get("branches", [])
        
        # If terminal node without branches
        if not branches:
            if not self.solved_status[node] and self.heuristics[node] == 0:
                self.solved_status[node] = True
            return False

        min_cost = float("inf")
        best_b = None
        is_solved = False
        branch_evaluations = []

        for branch in branches:
            b_cost, b_solved = self.calculate_branch_cost(branch)
            branch_evaluations.append({
                "type": branch.get("type", "OR"),
                "children": branch.get("nodes", []),
                "cost": b_cost,
                "all_children_solved": b_solved
            })

            if b_cost < min_cost:
                min_cost = b_cost
                best_b = branch
                is_solved = b_solved

        old_cost = self.heuristics[node]
        old_solved = self.solved_status[node]

        self.heuristics[node] = min_cost
        self.best_branch[node] = best_b
        self.solved_status[node] = is_solved

        # Record update step in audit trail
        self.audit_trail.append({
            "node": node,
            "previous_cost": old_cost,
            "updated_cost": min_cost,
            "chosen_branch": best_b,
            "solved": is_solved,
            "branch_evaluations": branch_evaluations,
            "reason": f"Recalculated minimum cost for node '{node}' among available branches."
        })

        return (old_cost != min_cost) or (old_solved != is_solved)

    def find_unsolved_leaf(self, node: str) -> Optional[str]:
        """
        Traverses the current best partial solution tree starting from `node`
        to find an unsolved non-terminal leaf node to expand next.
        """
        if self.solved_status.get(node, False):
            return None

        best_b = self.best_branch.get(node)
        if best_b is None:
            # If this node has branches to expand, return it as the leaf
            if self.graph.get(node, {}).get("branches"):
                return node
            return None

        # Recursively search in children of the current best branch
        for child in best_b.get("nodes", []):
            if not self.solved_status.get(child, False):
                leaf = self.find_unsolved_leaf(child)
                if leaf:
                    return leaf
                elif self.graph.get(child, {}).get("branches"):
                    return child

        return None

    def propagate_bottom_up(self, start_node: str):
        """
        Propagates cost updates bottom-up through ancestor nodes.
        """
        queue = [start_node]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            changed = self.update_node_cost(current)
            if changed:
                for parent in self.parents.get(current, []):
                    if parent not in visited:
                        queue.append(parent)

    def search(self, root: str, max_iterations: int = 50) -> Dict[str, Any]:
        """
        Runs the AO* search algorithm from the root node.

        Returns a dictionary with:
            - 'root': Starting root node
            - 'cost': Final optimal cost of the solution tree
            - 'solved': Boolean flag indicating if root solution was found
            - 'solution_tree': Selected best AND/OR sub-tree
            - 'audit_trail': Explainable step-by-step decision log
        """
        if root not in self.graph:
            raise ValueError(f"Root node '{root}' does not exist in graph.")

        iteration = 0
        while not self.solved_status[root] and iteration < max_iterations:
            iteration += 1

            # 1. Find the next unsolved leaf node in the current best partial tree
            leaf = self.find_unsolved_leaf(root)
            if leaf is None:
                # If no more unsolved leaves can be expanded, check root update
                self.update_node_cost(root)
                break

            # 2. Update and expand the leaf node, then propagate bottom-up
            self.propagate_bottom_up(leaf)

        solution_tree = self.extract_solution_tree(root)

        return {
            "root": root,
            "cost": self.heuristics[root],
            "solved": self.solved_status[root],
            "solution_tree": solution_tree,
            "audit_trail": self.audit_trail
        }

    def extract_solution_tree(self, node: str) -> Dict[str, Any]:
        """
        Recursively extracts the final best AND/OR solution sub-tree.
        """
        best_b = self.best_branch.get(node)
        if not best_b:
            return {
                "node": node,
                "cost": self.heuristics.get(node, 0),
                "solved": self.solved_status.get(node, False),
                "children": []
            }

        return {
            "node": node,
            "branch_type": best_b.get("type", "OR"),
            "cost": self.heuristics.get(node, 0),
            "solved": self.solved_status.get(node, False),
            "children": [self.extract_solution_tree(child) for child in best_b.get("nodes", [])]
        }


def ao_star_search(graph: Dict[str, Any], root: str) -> Dict[str, Any]:
    """
    Convenience function to run AO* search on an AND/OR graph.
    """
    solver = AOStarSearch(graph)
    return solver.search(root)


if __name__ == "__main__":
    # Sample AND/OR graph demonstration problem
    # Node A can be solved via:
    # 1. OR Branch -> Node B (edge cost: 1, B heuristic: 6) => 1 + 6 = 7
    # 2. AND Branch -> Nodes C and D (edge cost: 2, C h: 2, D h: 3) => 2 + (2 + 3) = 7
    sample_and_or_graph = {
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

    print("=== Running AO* Search on Sample AND/OR Graph ===")
    result = ao_star_search(sample_and_or_graph, "A")

    print(f"\nRoot Node       : {result['root']}")
    print(f"Total Cost      : {result['cost']}")
    print(f"Solved Status   : {result['solved']}")
    print(f"\nFinal Solution Tree Structure:")
    
    def print_tree(tree, indent=0):
        prefix = "  " * indent
        b_type = f" [{tree.get('branch_type')}]" if 'branch_type' in tree else ""
        print(f"{prefix}- Node: {tree['node']}{b_type} (cost={tree['cost']}, solved={tree['solved']})")
        for child in tree.get("children", []):
            print_tree(child, indent + 1)

    print_tree(result["solution_tree"])

    print(f"\n=== Audit Trail ({len(result['audit_trail'])} step(s)) ===")
    for idx, entry in enumerate(result["audit_trail"], 1):
        print(f"Step {idx}: Node '{entry['node']}' updated to cost {entry['updated_cost']} (solved={entry['solved']})")
        print(f"  Reason: {entry['reason']}")
        if entry.get("chosen_branch"):
            cb = entry["chosen_branch"]
            print(f"  Chosen Branch: {cb.get('type')} branch -> {cb.get('nodes')} (edge cost: {cb.get('cost')})")
