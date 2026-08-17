# Explainable AI Planner

## Overview
**Explainable AI Planner** is an educational and transparent decision-making framework that demonstrates explainable path planning using classical heuristic search algorithms: **A\*** (single-agent optimal pathfinding) and **AO\*** (heuristic search in AND/OR graphs with subproblem decomposition).

Every search decision, priority queue expansion, branch evaluation, cost update, and route recommendation is logged into a structured, step-by-step audit trail, providing complete algorithmic transparency.

---

## Features
- **A\* Shortest-Path Search**: Optimal, heuristic-guided graph traversal.
- **AO\* AND/OR Search**: Heuristic search across hypergraphs decomposing goals into sub-problems.
- **Robust Graph Validation**: Strict schema and integrity validation for nodes, weighted edges, and heuristics.
- **Explainable Audit Trails**: Comprehensive expansion logs detailing every decision rationale and neighbor comparison.
- **Deterministic Route Recommendation**: Explainable route advantage scoring without fake confidence or black-box heuristics.
- **Interactive Streamlit Interface**: Visual, web-based explorer with real-time parameter tweaking and expanders.
- **Graph Visualization**: Dynamic Graphviz rendering with optimal path highlighting.
- **AO\* Decision-Tree Visualization**: Visual representation of OR alternatives and AND hyper-branch junctions.

---

## Algorithms

### 1. A\* Search Algorithm
A\* determines the shortest path from a start node to a goal node by minimizing the total evaluation function $f(n)$:

$$f(n) = g(n) + h(n)$$

- **$g(n)$ (Actual Cost)**: The exact accumulated cost from the starting node to node $n$.
- **$h(n)$ (Heuristic Estimate)**: The estimated cost from node $n$ to the goal.
- **$f(n)$ (Total Estimated Cost)**: The projected total cost of a path passing through node $n$.

A\* maintains a priority queue of open frontier nodes, expanding the candidate with the lowest $f(n)$ score at each step.

### 2. AO\* Search Algorithm
AO\* solves complex planning tasks where goals can be achieved either through independent choices or by decomposing a task into multiple mandatory sub-goals:

- **OR Node / Branch**: Represents a choice where the planner selects **one alternative** with the lowest expected cost.
- **AND Node / Branch**: Represents a decomposition where the planner **must solve all required children** to resolve the parent goal. The branch cost is the sum of edge costs and child heuristics.
- **Bottom-Up Cost Propagation**: After expanding a leaf node, cost changes and `SOLVED` flags propagate upward to refine ancestor estimates until the root is solved.

---

## Project Structure

```text
expalinable ai planner/
├── data/
│   ├── graph.json            # Weighted graph and heuristics for A*
│   └── and_or_graph.json     # AND/OR graph structure for AO*
├── src/
│   ├── astar.py              # A* search with explainable audit trail
│   ├── ao_star.py            # AO* AND/OR search implementation
│   ├── graph_loader.py       # JSON graph loading and schema validation
│   ├── planner.py            # Unified interactive CLI planner interface
│   └── recommender.py        # Deterministic route advantage scoring
├── tests/
│   ├── test_astar.py         # Unit tests for A* search and audit trail
│   ├── test_ao_star.py       # Unit tests for AO* search and branches
│   └── test_recommender.py   # Unit tests for recommender confidence scoring
├── app.py                    # Streamlit visual web application
├── main.py                   # Main CLI entry point
├── requirements.txt          # Python dependency specifications
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "expalinable ai planner"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### 1. Interactive Streamlit Web Interface
Launch the interactive web UI:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Command-Line Entry Point
Run the default pipeline report:
```bash
python main.py
```

### 3. Interactive CLI Planner
Run the interactive console menu:
```bash
python -m src.planner
```

---

## Running Tests

Execute the complete automated unit test suite:
```bash
python -m unittest discover -s tests -v
```

---

## Example Results

### A\* Search (`data/graph.json`, Start: `A`, Goal: `G`)
- **Optimal Path**: `A -> C -> E -> G`
- **Total Cost**: `7`
- **Steps**: `3`
- **Confidence**: `0.22` *(The selected route costs 7, while the best known alternative costs 9, giving an advantage score of 0.22)*

### AO\* Search (`data/and_or_graph.json`, Root: `A`)
- **Optimal Solution Tree**:
  ```text
  - Node 'A' [OR] (Cost: 3.0, Status: Solved)
    - Node 'B' [OR] (Cost: 2.0, Status: Solved)
      - Node 'G1' (Cost: 0.0, Status: Solved)
  ```
- **Total Cost**: `3.0`
- **Solved Status**: `True`

---

## Explainability

The planner exposes human-interpretable reasoning at every stage:
- **Selected Nodes**: Exact identity of the node popped from the frontier and the reason why ($f$-score minimization).
- **Metric Breakdown**: Clear visibility of $g(n)$, $h(n)$, and $f(n)$ for every expansion.
- **Neighbor Decisions**: Explicit logging of every explored neighbor, distinguishing between `added to frontier`, `updated with a better cost`, and `skipped because existing path was better`.
- **Branch Decision Logic**: Detailed breakdown of why an OR branch was favored or how AND subproblems contributed to composite costs.
- **Deterministic Confidence**: Evaluates the competitive gap between the chosen route and the best available alternative without arbitrary heuristics.

---

## Limitations

- **Heuristic Quality**: The optimality and efficiency of A\* search depend directly on the admissibility and consistency of heuristic functions.
- **Evidence-Based Confidence**: Confidence is only calculated when reliable alternative routes are evaluated during search; otherwise, it explicitly reports `"Not enough evidence"`.
- **Scope**: Designed as an educational, explainable reference architecture rather than a high-throughput production routing engine.

---

## Future Improvements

- **Larger Benchmark Graphs**: Support for OpenStreetMap coordinates and grid-world benchmarks.
- **Interactive Graph Editor**: Visual UI to add/edit nodes, edge weights, and heuristics on the fly.
- **Additional Algorithms**: Integration of IDA\* (Iterative Deepening A\*), Bidirectional A\*, and SMA\*.
- **Natural Language Explanations**: Richer textual explanations using structured rule engines.
- **Real-World Constraints**: Dynamic edge weights, time windows, and multi-criteria optimization (e.g., cost vs. duration).
