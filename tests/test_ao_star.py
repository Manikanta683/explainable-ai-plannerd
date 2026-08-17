import sys
import unittest
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ao_star import AOStarSearch, ao_star_search


class TestAOStarSearch(unittest.TestCase):
    def setUp(self):
        # Standard example AND/OR graph
        self.example_graph = {
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

    def test_ao_star_finds_solution_for_example_graph(self):
        """1. AO* successfully finds a solution for the example AND/OR graph."""
        result = ao_star_search(self.example_graph, "A")
        self.assertTrue(result.get("solved", False))
        self.assertIsNotNone(result.get("solution_tree"))

    def test_result_structure(self):
        """2. The returned result contains solution_tree, total_cost (or cost), and audit_trail."""
        result = ao_star_search(self.example_graph, "A")
        self.assertIn("solution_tree", result)
        self.assertIn("audit_trail", result)
        self.assertTrue("total_cost" in result or "cost" in result)

    def test_total_cost_correctness(self):
        """3. total_cost is correct for the example graph (Cost = 3.0 via B -> G1)."""
        result = ao_star_search(self.example_graph, "A")
        total_cost = result.get("total_cost", result.get("cost"))
        self.assertEqual(total_cost, 3.0)

    def test_audit_trail_not_empty(self):
        """4. The audit_trail is not empty."""
        result = ao_star_search(self.example_graph, "A")
        self.assertGreater(len(result["audit_trail"]), 0)

    def test_audit_trail_entry_details(self):
        """5. Every audit trail entry contains expanded node, chosen branch, and cost calculation."""
        result = ao_star_search(self.example_graph, "A")
        for entry in result["audit_trail"]:
            self.assertIn("node", entry)
            self.assertIn("updated_cost", entry)
            self.assertIn("chosen_branch", entry)
            self.assertIn("reason", entry)

    def test_invalid_start_node_raises_value_error(self):
        """6. Test that an invalid start node raises ValueError."""
        with self.assertRaises(ValueError):
            ao_star_search(self.example_graph, "INVALID_NODE")

    def test_no_valid_solution_handling(self):
        """7. Test a graph where no valid solution exists."""
        unsolvable_graph = {
            "A": {
                "heuristic": 10,
                "branches": [
                    {"type": "OR", "nodes": ["B"], "cost": 2}
                ]
            },
            "B": {
                "heuristic": 5,
                "branches": [],
                "solved": False
            }
        }
        try:
            result = ao_star_search(unsolvable_graph, "A")
            self.assertFalse(result.get("solved", False))
        except ValueError:
            pass

    def test_and_branches_require_all_children_solved(self):
        """8. Verify that AND branches require all required child nodes to be solved."""
        graph = {
            "A": {
                "heuristic": 10,
                "branches": [
                    {"type": "AND", "nodes": ["B", "C"], "cost": 1}
                ]
            },
            "B": {"heuristic": 0, "branches": [], "solved": True},   # B is solved
            "C": {"heuristic": 5, "branches": [], "solved": False}  # C is NOT solved
        }
        solver = AOStarSearch(graph)
        branch = graph["A"]["branches"][0]
        cost, is_solved = solver.calculate_branch_cost(branch)
        # Cost = edge cost (1) + B(0) + C(5) = 6
        self.assertEqual(cost, 6)
        # Branch is NOT solved because C is unsolved
        self.assertFalse(is_solved)

        # Mark C as solved and recalculate
        solver.solved_status["C"] = True
        cost, is_solved = solver.calculate_branch_cost(branch)
        # Both are now solved, branch must be marked solved
        self.assertTrue(is_solved)

    def test_or_branches_choose_lowest_cost(self):
        """9. Verify that OR branches choose the lowest-cost available alternative."""
        graph = {
            "A": {
                "heuristic": 10,
                "branches": [
                    {"type": "OR", "nodes": ["Expensive"], "cost": 10},
                    {"type": "OR", "nodes": ["Cheap"], "cost": 2}
                ]
            },
            "Expensive": {"heuristic": 0, "branches": [], "solved": True},
            "Cheap": {"heuristic": 0, "branches": [], "solved": True}
        }
        result = ao_star_search(graph, "A")
        total_cost = result.get("total_cost", result.get("cost"))
        self.assertEqual(total_cost, 2.0)
        self.assertEqual(result["solution_tree"]["children"][0]["node"], "Cheap")


if __name__ == "__main__":
    unittest.main()
