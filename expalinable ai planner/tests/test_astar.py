import sys
import unittest
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.astar import astar_search
from src.graph_loader import load_graph


class TestAstarSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_path = PROJECT_ROOT / "data" / "graph.json"
        cls.graph = load_graph(cls.graph_path)

    def test_shortest_path_and_cost(self):
        """Verify A* finds the correct shortest path and cost from A to G."""
        result = astar_search(self.graph, "A", "G")
        expected_path = ["A", "C", "E", "G"]
        expected_cost = 7

        self.assertEqual(result["path"], expected_path)
        self.assertEqual(result["cost"], expected_cost)

    def test_audit_trail_structure(self):
        """Verify audit trail is not empty and each entry has required keys."""
        result = astar_search(self.graph, "A", "G")
        audit_trail = result.get("audit_trail", [])

        # Audit trail must not be empty
        self.assertTrue(len(audit_trail) > 0)

        required_keys = {"node", "g", "h", "f", "reason", "neighbors"}
        for entry in audit_trail:
            for key in required_keys:
                self.assertIn(key, entry, f"Missing key '{key}' in audit trail entry: {entry}")
            for neighbor in entry.get("neighbors", []):
                self.assertIn("node", neighbor)
                self.assertIn("tentative_g", neighbor)
                self.assertIn("action", neighbor)

    def test_invalid_start_node_raises_value_error(self):
        """Verify an invalid start node raises a ValueError."""
        with self.assertRaises(ValueError):
            astar_search(self.graph, "Z", "G")

    def test_invalid_goal_node_raises_value_error(self):
        """Verify an invalid goal node raises a ValueError."""
        with self.assertRaises(ValueError):
            astar_search(self.graph, "A", "Z")

    def test_no_possible_route_raises_value_error(self):
        """Verify a graph with no possible route from start to goal raises ValueError."""
        # In graph.json, node G has no outgoing edges, so G to A has no path
        with self.assertRaises(ValueError):
            astar_search(self.graph, "G", "A")

        # Test with a disconnected custom graph as well
        disconnected_graph = {
            "nodes": ["A", "B"],
            "edges": {"A": {}, "B": {}},
            "heuristics": {"A": 1, "B": 0}
        }
        with self.assertRaises(ValueError):
            astar_search(disconnected_graph, "A", "B")


if __name__ == "__main__":
    unittest.main()
