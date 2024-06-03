import os
from pathlib import Path
from typing import Any, List

from tqdm.auto import tqdm

from ollama_prompter.models.api.base_model import BaseModel
from ollama_prompter.prompter.prompter import Prompter
from ollama_prompter.prompter.prompt_cache import PromptCache


class Pipeline:

    def __init__(
        self, 
        prompters: List[Prompter], 
        model: BaseModel, 
        structured_output: bool = True, 
        **kwargs
    ):
        self.prompters = prompters
        self.model = model
        self.json_depth_limit: int = kwargs.get("json_depth_limit", 20)
        self.cache_prompt = kwargs.get("cache_prompt", True)
        self.cache_size = kwargs.get("cache_size", 200)
        self.prompt_cache = PromptCache(self.cache_size)
        self.conversation_path = kwargs.get("output_path", Path.cwd())
        self.structured_output = structured_output

        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[
            1 : self.model_args_count
        ]

        self.conversation_path = os.getcwd()
        self.model_dict = {
            key: value
            for key, value in model.__dict__.items()
            if is_string_or_digit(value)
        }

    def fit(self, text_input: str, **kwargs) -> Any:
        """
        Processes an input text through the pipeline:
         - Generate a prompt
         - Get a response from the model
         - Cache the response
         - Logs the conversation
         - Returns the output
        """
        outputs_list = []
        for prompter in tqdm(self.prompters):
            try:
                template, _ = prompter.generate(text_input, self.model.model_name, **kwargs)
            except ValueError as e:
                print(f"Error in generating prompt: {e}")
                return None

            if kwargs.get("verbose", False):
                print(template)

            output = self._get_output_from_cache_or_model(template)
            if output is None:
                return None

            outputs_list.append(output)

        return outputs_list

    def _get_output_from_cache_or_model(self, template):
        output = None

        if self.cache_prompt:
            output = self.prompt_cache.get(template)

        if output is None:
            try:
                response = self.model.execute_with_retry(prompt=template)
            except Exception as e:
                print(f"Error in model execution: {e}")
                return None
            # response = self.model.run(prompt=template)

            if self.structured_output:
                output = self.model.model_output(
                    response, json_depth_limit=self.json_depth_limit
                )
            else:
                output = response

            if self.cache_prompt:
                self.prompt_cache.add(template, output)

        return output
    

def is_string_or_digit(obj):
    return isinstance(obj, (str, int, float))