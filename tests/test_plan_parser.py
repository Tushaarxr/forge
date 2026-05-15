import pytest
import json
from pathlib import Path
from forge.plan_parser import parse_plan
from forge.auto_runner import AutoTask, PlanStatus

def test_parse_plan_plain_text(tmp_path):
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("1. Scaffold project\n2. Add models\n- Setup DB\nBlank line below\n\n# A comment\nCreate routes", encoding="utf-8")
    
    plan, is_raw = parse_plan(str(plan_file), goal="Test Plan")
    assert is_raw is True
    
    assert plan.goal == "Test Plan"
    assert len(plan.tasks) == 5
    assert plan.tasks[0].description == "Scaffold project"
    assert plan.tasks[1].description == "Add models"
    assert plan.tasks[2].description == "Setup DB"
    assert plan.tasks[3].description == "Blank line below"
    assert plan.tasks[4].description == "Create routes"

def test_parse_plan_json(tmp_path):
    plan_file = tmp_path / "plan.json"
    data = {
        "goal": "JSON Plan",
        "status": "pending",
        "created_at": "2026-01-01T00:00:00",
        "tasks": [
            {"id": 1, "description": "JSON task 1"},
            {"id": 2, "description": "JSON task 2"}
        ],
        "checkpoints": []
    }
    plan_file.write_text(json.dumps(data), encoding="utf-8")
    
    plan, is_raw = parse_plan(str(plan_file), goal="Overridden Goal")
    assert is_raw is False
    
    assert plan.goal == "Overridden Goal"
    assert len(plan.tasks) == 2
    assert plan.tasks[0].description == "JSON task 1"
    assert plan.tasks[1].description == "JSON task 2"

def test_parse_plan_raw_string():
    raw_plan = "1. Scaffold project\n2. Add models"
    plan, is_raw = parse_plan(raw_plan, goal="Raw Plan")
    assert is_raw is True
    
    assert plan.goal == "Raw Plan"
    assert len(plan.tasks) == 2
    assert plan.tasks[0].description == "Scaffold project"
    assert plan.tasks[1].description == "Add models"

def test_parse_plan_missing_file_fails():
    with pytest.raises(FileNotFoundError):
        parse_plan("non_existent_file.txt", goal="Empty")

def test_parse_plan_empty_fails():
    with pytest.raises(ValueError, match="No tasks could be parsed"):
        parse_plan("   \n# Just a comment\n", goal="Empty")


# Edge cases for plan parsing
def test_parse_invalid_json_fails(tmp_path):
    """Invalid JSON in plan file raises error."""
    plan_file = tmp_path / "plan.json"
    plan_file.write_text("{invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_plan(str(plan_file), goal="Test")


def test_parse_plan_malformed_json_structure(tmp_path):
    """JSON without expected structure raises appropriate error."""
    plan_file = tmp_path / "plan.json"
    plan_file.write_text('{"goal": "test"}', encoding="utf-8")  # Missing tasks key
    plan, is_raw = parse_plan(str(plan_file), goal="Test")
    # Should be treated as raw or empty
    assert plan.tasks == []


def test_parse_plan_with_whitespace_only(tmp_path):
    """Whitespace-only content raises ValueError."""
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("   \n\n\t\t  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="No tasks could be parsed"):
        parse_plan(str(plan_file), goal="Test")


def test_parse_plan_task_without_description(tmp_path):
    """Task entry without description uses default."""
    plan_file = tmp_path / "plan.json"
    data = {
        "goal": "Test",
        "status": "pending",
        "created_at": "2026-01-01T00:00:00",
        "tasks": [
            {"id": 1},  # Missing description
        ],
        "checkpoints": []
    }
    plan_file.write_text(json.dumps(data), encoding="utf-8")
    plan, is_raw = parse_plan(str(plan_file), goal="Test")
    assert len(plan.tasks) == 1
    # Description should default to empty or handle gracefully
    assert plan.tasks[0].description == ""


def test_parse_plan_mixed_numbering_styles(tmp_path):
    """Mixed numbering styles (1., 2-, 3.) all parsed."""
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("1. First task\n2- Second task\n3. Third task", encoding="utf-8")
    plan, is_raw = parse_plan(str(plan_file), goal="Test")
    assert len(plan.tasks) == 3


def test_parse_plan_unicode_content(tmp_path):
    """Unicode content in plan is preserved."""
    plan_file = tmp_path / "plan.txt"
    plan_file.write_text("1. Task with émoji 🚀\n2. 日本語タスク", encoding="utf-8")
    plan, is_raw = parse_plan(str(plan_file), goal="Test")
    assert "émoji" in plan.tasks[0].description
    assert "タスク" in plan.tasks[1].description
