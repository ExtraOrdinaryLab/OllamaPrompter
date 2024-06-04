import itertools
from typing import List, Dict, Mapping, Union, Iterator, Any

import ollama

from ollama_prompter.parser import Parser
from ollama_prompter.models.api.base_model import BaseModel


class Ollama(BaseModel):
    """
    Ollama API
    
    Note:
        Need to run local builds for Ollama to start the server first. Ollama has a REST API for running and managing models.

    Tip: You can even set up Ollama on remote server
        On your remote server, start by running `ollama serve` and run `ngrok http 11434`. 
        You will get a HTTP endpoint like "https://75fe5c3e.ngrok.io".
        Open your browser and input the HTTP endpoint. 
        If you see "Ollama is running", then it's working!
    """
    name = "Ollama"
    description = "Ollama for text completion using various models."

    def __init__(
        self, 
        api_key: str = None, 
        model_name: str = None, 
        endpoint: str = 'http://localhost:11434', 
        temperature: float = 0.7, 
        top_p: float = 1, 
        top_k: int = 1, 
        api_await: int = 60, 
        api_retry: int = 5, 
    ) -> None:
        self.endpoint = endpoint
        self._client = ollama.Client(host=self.endpoint)

        self.temperature = temperature # Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic.
        self.top_p = top_p # When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.
        self.top_k = top_k # When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.
        self.parameters = self.get_parameters()
        super().__init__(api_key, model_name, api_await, api_retry)

    def set_key(self, api_key: str):
        """
        Set endpoint API key if needed.
        """
        self.api_key = 'ollama'

    def _verify_model(self):
        """
        Verify the model is supported by the endpoint.
        """
        SUPPORTED_MODELS = {
            "chat_models": set(
                [model['name'] for model in self._client.list()['models']]
            )
        }
        if self.model_name not in SUPPORTED_MODELS["chat_models"]:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
    def supported_models(self) -> List[str]:
        SUPPORTED_MODELS = {
            "chat_models": set(
                [model['name'] for model in self._client.list()['models']]
            )
        }
        return list(itertools.chain(*SUPPORTED_MODELS.values()))

    def get_parameters(self):
        """
        Get model parameters.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
    
    def get_description(self):
        """
        Get model description.
        """
        return self.description
    
    def get_endpoint(self):
        """
        Get model endpoint.
        """
        return self.endpoint
    
    def set_model_name(self, model_name: str):
        self.model_name = model_name
        self._verify_model()

    def run(self, prompt: str) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
        """
        Run the LLM on the given prompt list.
        """
        prompt_template = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}, 
        ]
        self.parameters["messages"] = prompt_template
        response = self._client.chat(
            model=self.model_name, 
            messages=prompt_template, 
            options=self.parameters
        )
        return response
    
    def model_output_raw(self, response: Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]) -> Dict:
        data = {}
        content = str(response['message']['content'])
        data["text"] = content.strip()
        return data
    
    def model_output(
        self, 
        response: Union[Mapping[str, Any], Iterator[Mapping[str, Any]]], 
        json_depth_limit: int
    ) -> Dict:
        data = self.model_output_raw(response)
        # Try to parse the input JSON string and complete it if it is incomplete
        data["parsed"] = Parser().fit(data["text"], json_depth_limit)
        return data