import os
import json
from glob import glob
from typing import List, Union, Dict

from jinja2 import Template, Environment, FileSystemLoader, meta


class TemplateLoader:

    def __init__(self):
        self.loaded_templates = {}

    def load_template(
        self, template: str, model_name: str, from_string: bool = False
    ) -> Union[Dict[str, str]]:
        if template in self.loaded_templates:
            return self.loaded_templates[template]

        if from_string:
            template_instance = Template(template)
            template_data = {
                "template_name": "from_string",
                "template_dir": None,
                "environment": None,
                "template": template_instance,
            }
        else:
            template_data = self._load_template_from_path(template, model_name)

        self.loaded_templates[template] = template_data
        return self.loaded_templates[template]
    
    def _load_template_from_path(self, template: str, model_name: str) -> dict:
        """
        Load a Jinja2 template from the given path.
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        current_dir, _ = os.path.split(current_dir)
        templates_dir = os.path.join(current_dir, "prompts")
        all_folders = {
            f"{folder}.jinja": folder for folder in os.listdir(templates_dir)
        }

        if template in all_folders:
            meta_data = self._get_metadata(template, templates_dir, model_name)

            template_name = meta_data["metadata"]["file_name"]
            template_dir = meta_data["metadata"]["file_path"]
            environment = Environment(loader=FileSystemLoader(template_dir))
            template_instance = environment.get_template(template_name)

        else:
            self._verify_template_path(template)
            custom_template_dir, custom_template_name = os.path.split(template)

            template_name = custom_template_name
            template_dir = custom_template_dir
            environment = Environment(loader=FileSystemLoader(template_dir))
            template_instance = environment.get_template(custom_template_name)

        return {
            "template_name": template_name,
            "template_dir": template_dir,
            "environment": environment,
            "template": template_instance,
        }
    
    def _get_metadata(self, template_name: str, template_path: str, model_name: str):
        template_name, _ = template_name.split(".jinja")
        metadata_files = glob(
            os.path.join(template_path, template_name, "metadata.json")
        )

        metadata = read_json(metadata_files[0])
        metadata = self.search_model(metadata, model_name)
        metadata["file_path"] = os.path.join(template_path, template_name)
    
        return {"metadata": metadata}
    
    def search_model(self, data, model_name):
        all_models = []
        for sample in data:
            if model_name in sample['models']:
                return sample
            else:
                all_models.extend(sample['models'])
        raise ValueError(
            f"Model not found. Please choose the model from : {all_models}"
        )
    
    def _verify_template_path(self, templates_path: str):
        """
        Verify the existence of the template file.
        """
        if not os.path.isfile(templates_path):
            raise ValueError(f"Templates path {templates_path} does not exist")
        
    def list_templates(self, environment) -> List[str]:
        """
        List all templates in the specified environment.
        """
        return environment.list_templates()

    def get_template_variables(self, environment: Environment, template_name: str) -> List[str]:
        """
        Get a list of undeclared variables for the specified template.
        """
        template_source = environment.loader.get_source(environment, template_name)
        parsed_content = environment.parse(template_source)
        return list(meta.find_undeclared_variables(parsed_content))


def read_json(json_file):
    """
    Reads JSON data from a file and returns a Python object.
    """
    with open(json_file) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON data from file {json_file}: {str(e)}"
            )
    return data