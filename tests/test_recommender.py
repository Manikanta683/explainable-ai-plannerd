import sys
import unittest
from pathlib import Path

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.astar import astar_search
from src.graph_loader import load_graph
from src.recommender import recommend_route


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph_path = PROJECT_ROOT / "data" / "graph.json"
        cls.graph = load_graph(cls.graph_path)

    def test_recommend_route_with_search_audit(self):
        """Verify recommend_route calculates the deterministic advantage score when alternatives exist."""
        search_result = astar_search(self.graph, "A", "G")
        recommendation = recommend_route(search_result)

        self.assertEqual(recommendation["recommended_path"], ["A", "C", "E", "G"])
        self.assertEqual(recommendation["cost"], 7)
        self.assertEqual(recommendation["steps"], 3)
        self.assertIsNotNone(recommendation["confidence"])
        self.assertAlmostEqual(recommendation["confidence"], 0.2222, places=2)
        self.assertIn("advantage score", recommendation["confidence_explanation"])

    def test_recommend_route_no_alternative_returns_none_confidence(self):
        """Verify that when no alternative route is available, confidence is None."""
        input_data = {
            "path": ["A", "C", "E", "G"],
            "cost": 7,
            "audit_trail": []
        }
        recommendation = recommend_route(input_data)
        self.assertIsNone(recommendation["confidence"])
        self.assertIn("Not enough evidence", recommendation["confidence_explanation"])

    def test_missing_fields_raises_value_error(self):
        """Verify that missing required keys raise ValueError."""
        with self.assertRaises(ValueError):
            recommend_route({"path": ["A", "B"]})

        with self.assertRaises(ValueError):
            recommend_route("invalid_input")


if __name__ == "__main__":
    unittest.main()
