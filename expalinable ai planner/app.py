"""
Explainable AI Planner - Streamlit Web Application
An interactive, explainable visual interface for A* and AO* search algorithms.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import streamlit as st

# Ensure src directory is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_loader import load_graph
from src.astar import astar_search
from src.recommender import recommend_route
from src.ao_star import ao_star_search


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Explainable AI Planner",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_and_or_graph_file(path: Path) -> dict:
    """Helper to load and parse the AND/OR graph JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_graphviz_dot(graph: Dict[str, Any], path: Optional[List[str]] = None) -> str:
    """
    Generates a Graphviz DOT representation of the standard graph for A*.
    Highlights nodes and directed edges that belong to the optimal path if provided.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", {})
    heuristics = graph.get("heuristics", {})

    path_nodes = set(path) if path else set()
    path_edges = set()
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))

    dot = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="transparent", margin=0.2];',
        '  node [shape=circle, fontname="Helvetica-Bold", fontsize=12, style="filled", fillcolor="#f8f9fa", color="#495057", penwidth=1.5];',
        '  edge [fontname="Helvetica", fontsize=11, color="#6c757d", penwidth=1.2, arrowsize=0.8];'
    ]

    # Render Nodes with Heuristic values
    for node in nodes:
        h_val = heuristics.get(node, 0)
        label = f"{node}\\n(h={h_val})"

        if node in path_nodes:
            # Highlight node on the path
            dot.append(
                f'  "{node}" [label="{label}", fillcolor="#d1e7dd", color="#198754", fontcolor="#0f5132", penwidth=3.0];'
            )
        else:
            dot.append(f'  "{node}" [label="{label}"];')

    # Render Directed Edges with Costs
    for u, neighbors in edges.items():
        for v, weight in neighbors.items():
            if (u, v) in path_edges:
                # Highlight edge on the path
                dot.append(
                    f'  "{u}" -> "{v}" [label=" {weight} ", color="#198754", fontcolor="#0f5132", penwidth=3.2, arrowsize=1.0, weight=3];'
                )
            else:
                dot.append(f'  "{u}" -> "{v}" [label=" {weight} "];')

    dot.append("}")
    return "\n".join(dot)


def extract_solution_elements(solution_tree: Dict[str, Any]) -> Tuple[Set[str], Set[Tuple[str, Tuple[str, ...]]]]:
    """
    Recursively extracts the set of solution nodes and branch choices from the AO* solution tree.
    """
    sol_nodes: Set[str] = set()
    sol_branches: Set[Tuple[str, Tuple[str, ...]]] = set()

    def traverse(tree: Dict[str, Any]):
        if not tree:
            return
        node = tree.get("node")
        if node:
            sol_nodes.add(node)
        children = tree.get("children", [])
        if children:
            child_names = tuple(c.get("node") for c in children if c.get("node"))
            sol_branches.add((node, child_names))
            for child in children:
                traverse(child)

    traverse(solution_tree)
    return sol_nodes, sol_branches


def generate_ao_graphviz_dot(
    and_or_graph: Dict[str, Any],
    solution_tree: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates a Graphviz DOT representation of an AND/OR graph.
    Visualizes OR branches, AND branches (with junction nodes), heuristic values,
    solved goal nodes, and highlights the chosen solution tree branches.
    """
    sol_nodes, sol_branches = extract_solution_elements(solution_tree) if solution_tree else (set(), set())

    dot = [
        "digraph AO {",
        '  graph [rankdir=TB, bgcolor="transparent", margin=0.2];',
        '  node [fontname="Helvetica", fontsize=11, style="filled", fillcolor="#f8f9fa", color="#495057", penwidth=1.5];',
        '  edge [fontname="Helvetica", fontsize=10, color="#6c757d", penwidth=1.2, arrowsize=0.8];'
    ]

    # Render Nodes
    for node, data in and_or_graph.items():
        h_val = data.get("heuristic", 0)
        is_terminal_solved = data.get("solved", False) or (h_val == 0 and not data.get("branches"))
        
        if node in sol_nodes:
            status_tag = "\\n[Solved]" if is_terminal_solved else ""
            label = f"{node}\\n(h={h_val}){status_tag}"
            dot.append(
                f'  "{node}" [shape=circle, label="{label}", fillcolor="#d1e7dd", color="#198754", fontcolor="#0f5132", penwidth=2.5];'
            )
        elif is_terminal_solved:
            label = f"{node}\\n(Goal, h=0)"
            dot.append(
                f'  "{node}" [shape=doublecircle, label="{label}", fillcolor="#e2e3e5", color="#198754", fontcolor="#198754", penwidth=2.0];'
            )
        else:
            label = f"{node}\\n(h={h_val})"
            dot.append(f'  "{node}" [shape=circle, label="{label}"];')

    # Render Branches (OR and AND)
    for parent_node, data in and_or_graph.items():
        branches = data.get("branches", [])
        for b_idx, branch in enumerate(branches):
            b_type = branch.get("type", "OR")
            cost = branch.get("cost", 1)
            children = branch.get("nodes", [])
            child_tuple = tuple(children)

            is_selected_branch = (parent_node, child_tuple) in sol_branches

            if b_type == "OR":
                for child in children:
                    if is_selected_branch:
                        dot.append(
                            f'  "{parent_node}" -> "{child}" [label=" OR (cost: {cost}) ", color="#198754", fontcolor="#0f5132", penwidth=3.0, weight=3];'
                        )
                    else:
                        dot.append(
                            f'  "{parent_node}" -> "{child}" [label=" OR (cost: {cost}) "];'
                        )
            elif b_type == "AND":
                # Create an AND connector junction
                junction_id = f"and_{parent_node}_{b_idx}"
                if is_selected_branch:
                    dot.append(
                        f'  "{junction_id}" [shape=box, style="filled,rounded", label="AND", fontsize=9, fillcolor="#d1e7dd", color="#198754", fontcolor="#0f5132", penwidth=2.0, height=0.25, width=0.4];'
                    )
                    dot.append(
                        f'  "{parent_node}" -> "{junction_id}" [label=" cost: {cost} ", color="#198754", fontcolor="#0f5132", penwidth=3.0, weight=3];'
                    )
                    for child in children:
                        dot.append(
                            f'  "{junction_id}" -> "{child}" [color="#198754", penwidth=2.5, style="solid"];'
                        )
                else:
                    dot.append(
                        f'  "{junction_id}" [shape=box, style="filled,rounded", label="AND", fontsize=9, fillcolor="#fff3cd", color="#ffc107", fontcolor="#856404", height=0.25, width=0.4];'
                    )
                    dot.append(
                        f'  "{parent_node}" -> "{junction_id}" [label=" cost: {cost} "];'
                    )
                    for child in children:
                        dot.append(
                            f'  "{junction_id}" -> "{child}" [style="dashed"];'
                        )

    dot.append("}")
    return "\n".join(dot)


def format_solution_tree_md(tree: dict, depth: int = 0) -> str:
    """Recursively formats the AND/OR solution tree as clean Markdown."""
    indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * depth
    branch_badge = f" `{tree.get('branch_type')}`" if tree.get("branch_type") else ""
    status_badge = "✅ Solved" if tree.get("solved") else "⏳ Unsolved"
    cost = tree.get("cost", 0)
    
    current_line = f"{indent}🔹 **Node {tree['node']}**{branch_badge} &nbsp;|&nbsp; Cost: `{cost}` &nbsp;|&nbsp; Status: {status_badge}"
    lines = [current_line]
    
    for child in tree.get("children", []):
        lines.append(format_solution_tree_md(child, depth + 1))
    return "\n\n".join(lines)


def main():
    st.title("🧭 Explainable AI Planner")
    st.caption("Transparent, deterministic pathfinding and subproblem decomposition.")

    # Sidebar for Algorithm Selection
    st.sidebar.header("⚙️ Configuration")
    algorithm_choice = st.sidebar.radio(
        "Choose Search Algorithm:",
        ("A* Search", "AO* Search")
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Explainability Notice**:\n"
        "Every decision, heuristic estimation, and pruning action is logged in real-time."
    )

    # ---------------------------------------------------------
    # A* Search Flow
    # ---------------------------------------------------------
    if algorithm_choice == "A* Search":
        st.subheader("📍 A* Shortest Path Search")
        st.write("Finds the optimal path between two nodes using $f(n) = g(n) + h(n)$.")

        # Formula Explanation Card
        with st.container():
            st.markdown(
                r"""
                > **📐 A\* Formula**  
                > $$f(n) = g(n) + h(n)$$  
                > - **$g(n)$**: Cost from the start node to node $n$.  
                > - **$h(n)$**: Estimated heuristic cost from node $n$ to the goal.  
                > - **$f(n)$**: Total estimated path cost through node $n$.
                """
            )

        graph_path = PROJECT_ROOT / "data" / "graph.json"

        try:
            graph = load_graph(graph_path)
            nodes = graph.get("nodes", [])
        except Exception as e:
            st.error(f"Failed to load graph from `{graph_path}`: {e}")
            return

        col1, col2 = st.columns(2)
        with col1:
            default_start_idx = nodes.index("A") if "A" in nodes else 0
            start_node = st.selectbox("Select Start Node:", nodes, index=default_start_idx)
        with col2:
            default_goal_idx = nodes.index("G") if "G" in nodes else min(len(nodes) - 1, 1)
            goal_node = st.selectbox("Select Goal Node:", nodes, index=default_goal_idx)

        run_search = st.button("🚀 Run A*", type="primary")

        if run_search:
            try:
                # 1. Run A* Search
                search_result = astar_search(graph, start_node, goal_node)
                
                # 2. Get Deterministic Recommendation
                recommendation = recommend_route(search_result)
                path = recommendation["recommended_path"]

                st.markdown("---")
                st.markdown("### 🏆 Recommended Route")

                path_str = " ➔ ".join(path)
                st.success(f"**Recommended Path:** `{path_str}`")

                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric(label="Total Cost", value=recommendation["cost"])
                m_col2.metric(label="Steps (Edges)", value=recommendation["steps"])
                
                conf_val = recommendation.get("confidence")
                if conf_val is not None:
                    m_col3.metric(label="Confidence", value=f"{conf_val:.2f}")
                else:
                    m_col3.metric(label="Confidence", value="Not enough evidence")

                m_col4.metric(label="Nodes Expanded", value=len(search_result.get("audit_trail", [])))

                # Recommendation Reason & Confidence Explanation
                st.info(f"💡 **Recommendation Rationale:** {recommendation['recommendation_reason']}")
                if recommendation.get("confidence_explanation"):
                    st.caption(f"ℹ️ **Confidence Analysis:** {recommendation['confidence_explanation']}")

                # Graph Visualizer with Highlighted Path
                st.markdown("### 🗺️ Visual Graph Representation (Highlighted Path)")
                st.caption(f"Path `{path_str}` highlighted in green with edge costs and heuristic labels.")
                dot_source = generate_graphviz_dot(graph, path=path)
                st.graphviz_chart(dot_source, use_container_width=True)

                # Audit Trail Table
                st.markdown("### 📋 A* Expansion Audit Trail")
                audit_trail = search_result.get("audit_trail", [])
                audit_table_data = [
                    {
                        "Step": idx,
                        "Node": step["node"],
                        "g": step["g"],
                        "h": step["h"],
                        "f": step["f"],
                        "Reason": step["reason"]
                    }
                    for idx, step in enumerate(audit_trail, 1)
                ]
                st.dataframe(audit_table_data, use_container_width=True, hide_index=True)

                # Expandable Detailed Neighbor Decisions
                with st.expander("🔍 Detailed Neighbor Decisions by Expansion Step"):
                    for idx, step in enumerate(audit_trail, 1):
                        st.markdown(f"**Step {idx}: Selected Node `{step['node']}` (g={step['g']}, h={step['h']}, f={step['f']})**")
                        st.write(f"*{step['reason']}*")
                        neighbors = step.get("neighbors", [])
                        if neighbors:
                            for nbr in neighbors:
                                action_color = "🟢" if "added" in nbr['action'] else ("🔵" if "updated" in nbr['action'] else "⚪")
                                st.markdown(
                                    f"- {action_color} **Node `{nbr['node']}`**: Tentative cost $g = {nbr['tentative_g']}$ ➔ *{nbr['action']}*"
                                )
                        else:
                            st.caption("No neighbor evaluations (Goal reached or terminal node).")
                        if idx < len(audit_trail):
                            st.divider()

            except ValueError as val_err:
                st.error(f"Search Failure: {val_err}")
                st.markdown("### 🗺️ Graph Topology")
                st.graphviz_chart(generate_graphviz_dot(graph), use_container_width=True)
            except Exception as err:
                st.error(f"An unexpected error occurred: {err}")
        else:
            # Display Default Graph Topology before Search is triggered
            st.markdown("---")
            st.markdown("### 🗺️ Visual Graph Representation")
            st.caption("Initial graph topology showing directed edges with costs and node heuristics (h).")
            dot_source = generate_graphviz_dot(graph, path=None)
            st.graphviz_chart(dot_source, use_container_width=True)

    # ---------------------------------------------------------
    # AO* Search Flow
    # ---------------------------------------------------------
    elif algorithm_choice == "AO* Search":
        st.subheader("🌳 AO* AND/OR Graph Search")
        st.write("Solves complex problems by decomposing goals into AND/OR subproblems.")

        # Legend Card
        with st.container():
            st.markdown(
                """
                > **📌 Decision Legend**  
                > - **OR Branch**: Choose **one** alternative to solve the problem.  
                > - **AND Branch**: Solve **all required** child alternatives.  
                > - **Solved Node**: Reached a terminal goal state.
                """
            )

        and_or_graph_path = PROJECT_ROOT / "data" / "and_or_graph.json"

        try:
            and_or_graph = load_and_or_graph_file(and_or_graph_path)
            ao_nodes = list(and_or_graph.keys())
        except Exception as e:
            st.error(f"Failed to load AND/OR graph from `{and_or_graph_path}`: {e}")
            return

        col1, _ = st.columns([1, 1])
        with col1:
            default_root_idx = ao_nodes.index("A") if "A" in ao_nodes else 0
            start_node = st.selectbox("Select Root / Start Node:", ao_nodes, index=default_root_idx)

        run_ao = st.button("🚀 Run AO*", type="primary")

        if run_ao:
            try:
                # Run AO* Search
                ao_result = ao_star_search(and_or_graph, start_node)
                solution_tree = ao_result["solution_tree"]

                st.markdown("---")
                st.markdown("### 📊 AO* Explanation")

                # Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Root Node", ao_result["root"])
                m2.metric("Total Cost", f"{ao_result['cost']:.1f}")
                solved_text = "Solved" if ao_result["solved"] else "Unsolved"
                m3.metric("Solved Status", solved_text)

                if ao_result["solved"]:
                    st.success(f"🎯 **Goal Solved!** Optimal solution sub-tree successfully determined from node `{ao_result['root']}` with total cost `{ao_result['cost']:.1f}`.")
                else:
                    st.warning("⚠️ Could not completely solve all required sub-goals.")

                # Interactive Decision-Tree Visualizer
                st.markdown("### 🌲 Interactive AO* Decision-Tree Visualizer")
                st.caption("Selected solution branches and nodes are dynamically highlighted in green.")
                ao_dot_source = generate_ao_graphviz_dot(and_or_graph, solution_tree=solution_tree)
                st.graphviz_chart(ao_dot_source, use_container_width=True)

                # Explanatory Expander: How AO* made this decision
                with st.expander("💡 How AO* made this decision", expanded=True):
                    st.markdown(
                        """
                        - **OR Node Decision**: For OR nodes, AO* evaluates available alternative branches and selects the single option with the **lowest estimated cost**.
                        - **AND Node Decision**: For AND nodes, AO* must solve **all required children** to satisfy the subproblem decomposition. The branch cost is the sum of costs for all sub-branches.
                        - **Bottom-Up Propagation**: When a leaf node is expanded, costs and solved states are propagated upward to ancestor nodes, refining estimates until the root is solved.
                        """
                    )

                # Hierarchical Solution Tree Structure
                st.markdown("### 🌿 Optimal Solution Tree Hierarchy")
                tree_md = format_solution_tree_md(solution_tree)
                st.markdown(tree_md, unsafe_allow_html=True)

                # Step-by-step Audit Table
                st.markdown("### 📋 Step-by-Step Cost Propagation Audit Trail")
                ao_audit = ao_result.get("audit_trail", [])
                
                ao_table_data = []
                for idx, entry in enumerate(ao_audit, 1):
                    chosen = entry.get("chosen_branch")
                    if chosen:
                        branch_str = f"{chosen.get('type')} ➔ {chosen.get('nodes')} (cost: {chosen.get('cost')})"
                    else:
                        branch_str = "None"

                    ao_table_data.append({
                        "Step": idx,
                        "Node": entry.get("node"),
                        "Updated Cost": entry.get("updated_cost"),
                        "Solved": "Yes" if entry.get("solved") else "No",
                        "Selected Branch": branch_str,
                        "Reason": entry.get("reason")
                    })

                st.dataframe(ao_table_data, use_container_width=True, hide_index=True)

                # Detailed Step Expanders
                with st.expander("🔍 Detailed Branch Evaluations by Step"):
                    for idx, entry in enumerate(ao_audit, 1):
                        solved_flag = "✅ Solved" if entry.get("solved") else "⏳ Open"
                        st.markdown(f"**Step {idx}: Node `{entry['node']}` (Updated Cost: {entry['updated_cost']}, Status: {solved_flag})**")
                        st.write(f"*{entry['reason']}*")
                        
                        chosen = entry.get("chosen_branch")
                        if chosen:
                            st.markdown(
                                f"- **Selected Branch:** `{chosen.get('type')}` branch ➔ "
                                f"Children: `{chosen.get('nodes')}` (Edge Cost: `{chosen.get('cost')}`)"
                            )

                        evaluations = entry.get("branch_evaluations", [])
                        if evaluations:
                            st.write("**Branch Evaluations:**")
                            for b_eval in evaluations:
                                b_solved_icon = "✅" if b_eval['all_children_solved'] else "⏳"
                                st.markdown(
                                    f"  - `{b_eval['type']}` branch to `{b_eval['children']}`: "
                                    f"Calculated Cost = `{b_eval['cost']}` ({b_solved_icon} Children Solved: {b_eval['all_children_solved']})"
                                )
                        if idx < len(ao_audit):
                            st.divider()

            except ValueError as val_err:
                st.error(f"Search Failure: {val_err}")
                st.markdown("### 🌲 AND/OR Graph Topology")
                st.graphviz_chart(generate_ao_graphviz_dot(and_or_graph), use_container_width=True)
            except Exception as err:
                st.error(f"An unexpected error occurred: {err}")
        else:
            # Display Default AND/OR Graph Topology before Search is triggered
            st.markdown("---")
            st.markdown("### 🌲 AND/OR Graph Representation")
            st.caption("Initial AND/OR graph showing OR/AND branches, heuristics (h), and terminal goal nodes.")
            ao_dot_source = generate_ao_graphviz_dot(and_or_graph, solution_tree=None)
            st.graphviz_chart(ao_dot_source, use_container_width=True)


if __name__ == "__main__":
    main()
