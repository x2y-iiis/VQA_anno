"""Prompt rendering backed by selectable package or user resource files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import PipelineDataError


_RESOURCE_PACKAGE = "las_annotation.resources"
_BUILTIN_PROFILES = {
    "short": "skills.json",
    "66": "skills_66.json",
    "long": "skills_86.json",
    "27": "skills.json",
    "86": "skills_86.json",
}
_HUMAN_SKILL_REWRITE_VERSION = "human-skill-terms/v1"


@dataclass(frozen=True)
class SkillVocabulary:
    name: str
    source: str
    digest: str
    items: tuple[tuple[str, str], ...]


def load_vocabulary(
    profile: str = "66",
    skills_file: str | Path | None = None,
) -> SkillVocabulary:
    """Load a built-in profile or an arbitrary vocabulary JSON file."""
    if skills_file is not None:
        path = Path(skills_file).expanduser().resolve()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise PipelineDataError(f"Cannot read skills file {path}: {exc}") from exc
        name = f"custom:{path.name}"
        source = str(path)
    else:
        try:
            resource_name = _BUILTIN_PROFILES[profile]
        except KeyError as exc:
            choices = ", ".join(("short", "66", "long"))
            raise PipelineDataError(
                f"Unknown skill profile {profile!r}; expected one of: {choices}."
            ) from exc
        text = _read_resource(resource_name)
        name = {
            "skills.json": "short",
            "skills_66.json": "66",
            "skills_86.json": "long",
        }[resource_name]
        source = f"package:{resource_name}"

    items = _parse_skills_json(text, source)
    canonical = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return SkillVocabulary(name=name, source=source, digest=digest, items=items)


def load_skills(
    profile: str = "66",
    skills_file: str | Path | None = None,
) -> tuple[tuple[str, str], ...]:
    """Compatibility helper returning only vocabulary items."""
    return load_vocabulary(profile, skills_file).items


def build_prompt_context(
    objects: Sequence[Mapping[str, Any]],
    vocabulary: SkillVocabulary | None = None,
    *,
    skill_profile: str = "66",
    skills_file: str | Path | None = None,
    embodiment: str = "robot",
    has_wrist_view: bool = True,
    task_instruction: str | None = None,
) -> str:
    if embodiment not in {"robot", "human"}:
        raise PipelineDataError("embodiment must be either 'robot' or 'human'.")
    vocabulary = vocabulary or load_vocabulary(skill_profile, skills_file)
    intro = (
        "context_intro.txt"
        if has_wrist_view
        else "single_head_context_intro.txt"
    )
    lines = []
    if not has_wrist_view:
        instruction = _required_text(task_instruction, "task_instruction")
        lines.extend(("【任务名称】", instruction, ""))
    lines.append(_load_text_resource(intro))
    if embodiment == "human":
        lines.extend(("", _load_text_resource("human_actor_rules.txt")))
    if objects:
        for index, item in enumerate(objects):
            category = _required_text(item.get("category"), f"objects[{index}].category")
            name = _required_text(item.get("name"), f"objects[{index}].name")
            description = _required_text(
                item.get("description"), f"objects[{index}].description"
            )
            lines.append(f"- {category}: {name}（{description}）")
    elif has_wrist_view:
        lines.append("- 未识别到明确的主要交互物品。")

    skill_lines = [
        f"【动作类别词表】每段的 `skill` 取值必须从以下 {len(vocabulary.items)} 类"
        "原子动作中选取一个（区分大小写，原样输出英文标签）："
    ]
    skill_lines.extend(
        f"- {skill}（{_skill_description(description, embodiment)}）"
        for skill, description in vocabulary.items
    )
    return (
        "\n".join(lines)
        + "\n\n"
        + "\n".join(skill_lines)
        + "\n\n"
        + _load_text_resource("skill_rules.txt")
    )


def prompt_policy_digest(embodiment: str, has_wrist_view: bool) -> str:
    """Fingerprint fixed prompt resources and embodiment-specific rewrites."""
    if embodiment not in {"robot", "human"}:
        raise PipelineDataError("embodiment must be either 'robot' or 'human'.")
    intro = "context_intro.txt" if has_wrist_view else "single_head_context_intro.txt"
    parts = [_read_resource(intro), _read_resource("skill_rules.txt")]
    if embodiment == "human":
        parts.extend((
            _read_resource("human_actor_rules.txt"),
            _HUMAN_SKILL_REWRITE_VERSION,
        ))
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _skill_description(description: str, embodiment: str) -> str:
    if embodiment != "human":
        return description
    return description.replace("夹爪无开合", "手的抓握状态不变").replace(
        "夹爪", "手"
    )


@lru_cache(maxsize=None)
def _read_resource(name: str) -> str:
    return files(_RESOURCE_PACKAGE).joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=2)
def _load_text_resource(name: str) -> str:
    text = _read_resource(name).strip()
    if not text:
        raise PipelineDataError(f"Resource {name} cannot be empty.")
    return text


def _parse_skills_json(text: str, source: str) -> tuple[tuple[str, str], ...]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PipelineDataError(f"Invalid skill JSON in {source}: {exc.msg}") from exc
    if not isinstance(value, list):
        raise PipelineDataError(f"Skill resource {source} must contain a JSON list.")

    skills: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise PipelineDataError(f"{source}[{index}] must be an object.")
        skill = _required_text(item.get("skill"), f"{source}[{index}].skill")
        description = _required_text(
            item.get("description"), f"{source}[{index}].description"
        )
        if skill in seen:
            raise PipelineDataError(f"Duplicate skill {skill!r} in {source}.")
        seen.add(skill)
        skills.append((skill, description))
    if not skills:
        raise PipelineDataError(f"Skill resource {source} cannot be empty.")
    return tuple(skills)


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineDataError(f"{label} must be a non-empty string.")
    return value.strip()
