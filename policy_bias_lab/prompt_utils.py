"""Task-neutral loading of policy-bias prompt templates."""

from pathlib import Path
from string import Template


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def prompt_template(name: str) -> Template:
    """Load one prompt template by filename from the package prompt directory."""
    return Template((PROMPTS_DIR / name).read_text())
