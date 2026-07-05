from __future__ import annotations

import json
from pathlib import Path

from xrdviz.models import ProjectState, project_from_dict, project_to_dict


def save_project(state: ProjectState, path: str | Path) -> None:
    output = Path(path)
    output.write_text(json.dumps(project_to_dict(state), indent=2), encoding="utf-8")


def load_project(path: str | Path) -> ProjectState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return project_from_dict(data)
