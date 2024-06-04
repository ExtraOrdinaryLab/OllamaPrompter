import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from jinja2 import Template, Environment, FileSystemLoader, meta


class Prompter(object):

    def __init__(self, template_name: str, template_dir: str) -> None:
        self.template_name = template_name
        self.template_dir = template_dir

    def generate(self, text: str, **kwargs) -> str:
        """
        Generates a prompt based on a template and input variables.
        """
        kwargs['text'] = text.strip()
        template_content = read_template(self.template_name, self.template_dir)
        environment = Environment(loader=FileSystemLoader(self.template_dir))
        template = environment.get_template(self.template_name)
        prompt = template.render(**kwargs)
        return prompt


def read_template(template_name: str, template_dir: str) -> str:
    """Read a template"""

    path = os.path.join(template_dir, template_name)

    if not os.path.exists(path):
        raise ValueError(f"{path} is not a valid template.")

    return Path(path).read_text()
