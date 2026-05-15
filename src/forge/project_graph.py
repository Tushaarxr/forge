"""Project graph management using NetworkX for dependency analysis."""

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import networkx as nx


logger = logging.getLogger(__name__)

GRAPH_PATH = os.getenv("GRAPH_PATH", ".forge/project_graph.json")


class ProjectGraph:
    """NetworkX-based project dependency graph manager."""

    def __init__(self) -> None:
        """Initialize the ProjectGraph with empty network."""
        self.graph: nx.DiGraph = nx.DiGraph()

    def _get_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
        }
        return ext_map.get(Path(file_path).suffix.lower(), "unknown")

    def _get_node_data(self, node_type: str, **kwargs: Any) -> dict[str, Any]:
        """Create node data dictionary based on type."""
        if node_type == "file":
            return {
                "path": kwargs["path"],
                "language": kwargs.get("language", ""),
                "last_modified": kwargs.get("last_modified", 0),
                "summary": "",
            }
        elif node_type == "function":
            return {
                "name": kwargs["name"],
                "file_path": kwargs["file_path"],
                "line_start": kwargs.get("line_start", 0),
                "line_end": kwargs.get("line_end", 0),
                "signature": kwargs.get("signature", ""),
            }
        elif node_type == "class":
            return {
                "name": kwargs["name"],
                "file_path": kwargs["file_path"],
                "line_start": kwargs.get("line_start", 0),
            }
        return {}

    def parse_file(self, file_path: str) -> None:
        """Parse a single file and add nodes/edges to the graph.

        Handles malformed files with try/except and logging.
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return

            lang = self._get_language(file_path)
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            mtime = os.path.getmtime(file_path)

            file_node = {
                "id": file_path,
                "type": "file",
                **self._get_node_data("file", path=file_path, language=lang, last_modified=mtime),
            }
            self.graph.add_node(file_path, **file_node)

            if lang == "python":
                try:
                    tree = ast.parse(content)
                    self._parse_python_ast(tree, file_path)
                except SyntaxError as e:
                    logger.warning(f"Syntax error in {file_path}: {e}")
            else:
                self._parse_regex_fallback(content, file_path)

        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")

    def _parse_python_ast(self, tree: ast.AST, file_path: str) -> None:
        """Parse Python AST to extract imports, functions, classes and calls."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.asname or alias.name
                    self._add_import_edge(file_path, target)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Use just the top-level module name for edge resolution
                top_module = module.split(".")[0] if module else ""
                if top_module:
                    self._add_import_edge(file_path, top_module)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_data = {
                    "id": f"{file_path}:{node.name}",
                    "type": "function",
                    **self._get_node_data(
                        "function",
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                        signature=f"def {node.name}(...)",
                    ),
                }
                func_node_id = f"{file_path}:{node.name}"
                self.graph.add_node(func_node_id, **func_data)
                self.graph.add_edge(file_path, func_node_id, type="contains")

            elif isinstance(node, ast.ClassDef):
                class_data = {
                    "id": f"{file_path}:{node.name}",
                    "type": "class",
                    **self._get_node_data(
                        "class",
                        name=node.name,
                        file_path=file_path,
                        line_start=node.lineno,
                    ),
                }
                class_node_id = f"{file_path}:{node.name}"
                self.graph.add_node(class_node_id, **class_data)
                self.graph.add_edge(file_path, class_node_id, type="contains")

    def _add_import_edge(self, source_file: str, module_name: str) -> None:
        """Add an import edge from source_file to the module.

        Tries to resolve the module name to an actual file node in the graph.
        If not found, still adds the edge to a placeholder node for graph completeness.
        """
        # Try to find a matching file node (e.g., module "utils" -> "utils.py")
        target_node = None
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if node_data.get("type") != "file":
                continue
            node_path = Path(node)
            # Match by stem (filename without extension)
            if node_path.stem == module_name:
                target_node = node
                break

        if target_node:
            self.graph.add_edge(source_file, target_node, type="imports")
        else:
            # Add a placeholder for the external module so the edge is recorded
            placeholder = f"__module__{module_name}__"
            if placeholder not in self.graph.nodes():
                self.graph.add_node(placeholder, type="external_module", name=module_name)
            self.graph.add_edge(source_file, placeholder, type="imports")

    def _parse_regex_fallback(self, content: str, file_path: str) -> None:
        """Regex fallback for non-Python files (.js/.ts/.go/.rs)."""
        # Extract require/import lines
        import_patterns = [
            r"(?:import|require)\s+['\"]([^'\"]+)['\"]",  # JS/TS require/import
            r"import\s+\"([^\"]+)\"",                       # Go import
            r"use\s+([\w:]+);",                             # Rust use
        ]
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                module_name = match.group(1).split("/")[-1].replace(".js", "").replace(".ts", "")
                self._add_import_edge(file_path, module_name)

        # Extract function/def/func definitions
        func_pattern = r"(?:^|\s)(?:def|func|function|fn)\s+(\w+)\s*\("
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            name = match.group(1)
            line = content[: match.start()].count("\n") + 1
            node_id = f"{file_path}:{name}"
            data = {
                "id": node_id,
                "type": "function",
                **self._get_node_data("function", name=name, file_path=file_path, line_start=line),
            }
            self.graph.add_node(node_id, **data)
            self.graph.add_edge(file_path, node_id, type="contains")

    def parse_project(self, root_dir: str) -> dict[str, int]:
        """Walk project directory and parse all supported files."""
        exclude_dirs = {".git", "__pycache__", "node_modules", ".venv", "dist", "build", ".forge"}

        files_parsed = 0
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Prune excluded directories in-place
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if ext in {".py", ".js", ".ts", ".go", ".rs"}:
                    file_path = os.path.join(dirpath, filename)
                    self.parse_file(file_path)
                    files_parsed += 1

        return {
            "files_parsed": files_parsed,
            "nodes_total": len(self.graph.nodes()),
            "edges_total": len(self.graph.edges()),
        }

    def get_edges(self) -> list[dict[str, str]]:
        """Return list of all edges as dicts with source/target/type keys."""
        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({"source": src, "target": tgt, "type": data.get("type", "")})
        return edges

    def get_affected(self, file_path: str, depth: int = 2) -> list[str]:
        """Get files that depend on or import the given file (reverse dependencies)."""
        affected = set()
        frontier = {file_path}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for pred in self.graph.predecessors(node):
                    if pred != file_path and pred not in affected:
                        # Only return actual file nodes
                        if self.graph.nodes[pred].get("type") == "file":
                            next_frontier.add(pred)
            affected.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return sorted(affected)

    def get_context_files(self, file_path: str, depth: int = 1) -> list[str]:
        """Get files that the given file imports (dependencies, forward edges)."""
        visited: set[str] = set()
        current_nodes = [file_path]

        for _ in range(depth):
            next_nodes: set[str] = set()
            for node in current_nodes:
                for succ in self.graph.successors(node):
                    if succ not in visited and succ != file_path:
                        if self.graph.nodes[succ].get("type") == "file":
                            next_nodes.add(succ)
            if not next_nodes:
                break
            visited.update(next_nodes)
            current_nodes = list(next_nodes)

        return sorted(visited)

    def get_summary(self) -> dict[str, Any]:
        """Return project graph summary statistics."""
        file_nodes = [n for n in self.graph.nodes() if self._is_file_node(n)]
        func_nodes = [n for n in self.graph.nodes() if self._is_function_node(n)]
        class_nodes = [n for n in self.graph.nodes() if self._is_class_node(n)]

        degrees = dict(self.graph.degree())
        sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        most_connected = [item[0] for item in sorted_degrees[:5]]

        return {
            "total_files": len(file_nodes),
            "total_functions": len(func_nodes),
            "total_classes": len(class_nodes),
            "total_edges": len(self.graph.edges()),
            "most_connected": most_connected,
        }

    def _is_file_node(self, node: str) -> bool:
        # FIX: Add existence check to prevent KeyError
        return node in self.graph and self.graph.nodes[node].get("type") == "file"

    def _is_function_node(self, node: str) -> bool:
        # FIX: Add existence check to prevent KeyError
        return node in self.graph and self.graph.nodes[node].get("type") == "function"

    def _is_class_node(self, node: str) -> bool:
        # FIX: Add existence check to prevent KeyError
        return node in self.graph and self.graph.nodes[node].get("type") == "class"

    def save(self) -> None:
        """Save graph to file using node-link data format."""
        Path(GRAPH_PATH).parent.mkdir(parents=True, exist_ok=True)
        # networkx >= 3.4 requires edges="edges" kwarg
        try:
            data = nx.node_link_data(self.graph, edges="edges")
        except TypeError:
            # Older networkx versions
            data = nx.node_link_data(self.graph)
        Path(GRAPH_PATH).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Graph saved to {GRAPH_PATH}")

    def load(self) -> None:
        """Load graph from file using node-link data format."""
        if not os.path.exists(GRAPH_PATH):
            logger.warning(f"{GRAPH_PATH} not found, starting with empty graph")
            return
        try:
            data = json.loads(Path(GRAPH_PATH).read_text(encoding="utf-8"))
            try:
                self.graph = nx.node_link_graph(data, edges="edges")
            except TypeError:
                self.graph = nx.node_link_graph(data)
            logger.info(f"Graph loaded from {GRAPH_PATH}")
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
