import os
from typing import List, Dict, Any, Optional

from ollama_prompter.prompter.template_loader import TemplateLoader


class Prompter:

    def __init__(
        self, 
        template, 
        from_string = False,
        allowed_missing_variables: Optional[List[str]] = None,
        default_variable_values: Optional[Dict[str, Any]] = None,
    ):
        self.template = template
        self.template_loader = TemplateLoader()
        self.allowed_missing_variables = [
            "examples",
            "description",
            "output_format",
        ]
        self.allowed_missing_variables.extend(allowed_missing_variables or [])
        self.default_variable_values = default_variable_values or {}
        self.from_string = from_string

    def update_default_variable_values(self, new_defaults: Dict[str, Any]) -> None:
        self.default_variable_values.update(new_defaults)

    def generate(self, text_input, model_name, **kwargs) -> str:
        """
        Generates a prompt based on a template and input variables.
        """
        loader = self.template_loader.load_template(
            self.template, model_name, self.from_string
        )

        kwargs["text_input"] = text_input

        if loader["environment"]:
            variables = self.template_loader.get_template_variables(
                loader["environment"], loader["template_name"]
            )
            variables_dict = {
                temp_variable_: kwargs.get(temp_variable_, None)
                for temp_variable_ in variables
            }

            variables_missing = [
                variable
                for variable in variables
                if variable not in kwargs
                and variable not in self.allowed_missing_variables
                and variable not in self.default_variable_values
            ]

            if variables_missing:
                raise ValueError(
                    f"Missing required variables in template {', '.join(variables_missing)}"
                )
        else:
            variables_dict = {"data": None}

        kwargs.update(self.default_variable_values)
        prompt = loader["template"].render(**kwargs).strip()

        if kwargs.get("verbose", False):
            print(prompt)

        return prompt, variables_dict