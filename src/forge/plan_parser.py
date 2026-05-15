import json
import re
import sys
from pathlib import Path
from typing import Any

from forge.auto_runner import AutoPlan, AutoTask


def parse_plan(source: str, goal: str = "") -> AutoPlan:
    """
    Accept:
    1. JSON file path (.json)
    2. Plain text file path (.txt, .md)
    3. "-" for stdin
    4. Raw string passed via --plan-text

    Plain text format (very forgiving parser):
      - Numbered lists:   "1. Create models.py"
      - Bullet lists:     "- Create models.py" or "* Create models.py"  
      - Bare lines:       "Create models.py" (each line = one task)
      - Blank lines ignored
      - Lines starting with # ignored (comments)
    
    Returns an AutoPlan object.
    """
    content = ""
    
    if source == "-":
        content = sys.stdin.read()
    elif "\n" in source:
        # Raw string with newlines is always treated as content
        content = source
    else:
        path = Path(source)
        if not path.exists():
            # Fuzzy match: try adding common extensions
            for ext in [".txt", ".md", ".json", ".txt.txt"]:
                candidate = Path(str(path) + ext)
                if candidate.exists():
                    from rich.console import Console
                    Console().print(f"[yellow]⚠️ Warning: '{source}' not found. Using '{candidate.name}' instead.[/yellow]")
                    path = candidate
                    break

        if path.exists():
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    plan = AutoPlan.from_dict(data)
                    if goal:
                        plan.goal = goal
                    return plan, False
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON plan file: {e}")
            else:
                content = path.read_text(encoding="utf-8")
        else:
            # File not found and no newlines in source string
            raise FileNotFoundError(
                f"Plan file not found: {source}\n"
                "Tip: Check for hidden file extensions (e.g. 'tasks.txt.txt') or verify your current directory."
            )
            
    tasks: list[AutoTask] = []
    
    # Parse text format
    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Strip numbering (e.g. "1. ", "12) ") or bullets ("- ", "* ")
        match = re.match(r"^(\d+[\.\)]\s+|[-*]\s+)?(.*)", line)
        if match:
            desc = match.group(2).strip()
            if desc:
                # Add task
                tasks.append(
                    AutoTask(
                        id=len(tasks) + 1,
                        description=desc,
                        active_file=None,
                        category="logic",
                        estimated_lines=0,
                        depends_on=[]
                    )
                )

    if not tasks:
        raise ValueError("No tasks could be parsed from the provided plan source.")
        
    plan = AutoPlan(
        goal=goal or "Custom Plan",
        created_at="", 
    )
    plan.tasks = tasks
    return plan, True
