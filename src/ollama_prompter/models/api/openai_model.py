import itertools
from typing import List, Dict, Tuple, Union, Optional, Any

import openai
import tiktoken
from openai.types.chat import ChatCompletion

from ollama_prompter.parser import Parser
from ollama_prompter.models.api.base_model import BaseModel


class OpenAI(BaseModel):

    name = "OpenAI"
    description = "OpenAI API for text completion using various models."

    SUPPORTED_MODELS = {
        "chat_models": set(
            [
                "gpt-3.5-turbo-0125", # The latest GPT-3.5 Turbo model
                "gpt-3.5-turbo", # Currently points to gpt-3.5-turbo-0125
                "gpt-3.5-turbo-1106", # GPT-3.5 Turbo model with improved instruction following
                "gpt-3.5-turbo-instruct", # Similar capabilities as GPT-3 era models
                "gpt-4-turbo", # The latest GPT-4 Turbo model with vision capabilities
                "gpt-4-0125-preview", # GPT-4 Turbo preview model intended to reduce cases of “laziness” 
                "gpt-4-turbo-preview", # Currently points to gpt-4-0125-preview, 
                "gpt-4", # Currently points to gpt-4-0613
                "gpt-4-0613", # Snapshot of gpt-4 from June 13th 2023 with improved function calling support
            ]
        )
    }

    def __init__(
        self, 
        api_key: str, 
        model_name: str, 
        temperature: float = 0.7, 
        top_p: float = 1, 
        n: int = 1, 
        stop: Optional[Union[str, List[str]]] = None, 
        logprobs: bool = False, 
        top_logprobs: int = None, 
        presence_penalty: float = 0, 
        frequency_penalty: float = 0, 
        logit_bias: Optional[Dict[str, int]] = None, 
        request_timeout: Union[float, Tuple[float, float]] = None, 
        api_await: int = 60, 
        api_retry: int = 5, 
    ) -> None:
        super().__init__(api_key, model_name, api_await, api_retry)
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias or {}
        self.request_timeout = request_timeout
        self._initialize_encoder()
        self.parameters = self.get_parameters()

    def set_key(self, api_key: str):
        """
        Set endpoint API key if needed.
        """
        self._openai = openai
        self._openai.api_key = api_key

    def _verify_model(self):
        """
        Verify the model is supported by the endpoint.
        """
        if self.model_name not in self.SUPPORTED_MODELS["chat_models"]:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
    def supported_models(self) -> List[str]:
        return list(itertools.chain(*self.SUPPORTED_MODELS.values()))

    def get_parameters(self):
        """
        Get model parameters.
        """
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stop": self.stop, 
            "logprobs": self.logprobs, 
            "top_logprobs": self.top_logprobs, 
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "timeout": self.request_timeout,
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
        model = self._openai.models.retrieve(self.model_name)
        return model["id"]
    
    def set_model_name(self, model_name: str):
        self.model_name = model_name
        self._verify_model()
    
    def run(self, prompt: str) -> ChatCompletion:
        """
        Run the LLM on the given prompt list.
        """
        client = self._openai.OpenAI(api_key=self.api_key)
        prompt_template = [
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}, 
        ]
        # https://community.openai.com/t/confused-about-max-tokens-parameter-with-gtp4-turbo-128k-tokenusedforprompt-or-4k/506681/2
        # self.parameters["max_tokens"] = self._calculate_max_tokens(prompt_template)
        self.parameters["messages"] = prompt_template
        response = client.chat.completions.create(
            model=self.model_name, 
            **self.parameters,
        )
        return response

    def _calculate_max_tokens(self, prompt: str) -> int:
        prompt_tokens = len(self.encoder.encode(str(prompt)))
        max_tokens = self._default_max_tokens(self.model_name) - prompt_tokens
        return max_tokens

    def _default_max_tokens(self, model_name: str) -> int:
        token_dict = {
            "gpt-3.5-turbo-0125": 16_385, 
            "gpt-3.5-turbo": 16_385, 
            "gpt-3.5-turbo-1106": 16_385, 
            "gpt-3.5-turbo-instruct": 4_096, 
            "gpt-4-turbo": 128_000, 
            "gpt-4-0125-preview": 128_000, 
            "gpt-4-turbo-preview": 128_000, 
            "gpt-4": 8_192, 
            "gpt-4-0613": 8_192,
        }
        return token_dict[model_name]
    
    def model_output(self, response: ChatCompletion, json_depth_limit: int) -> Dict:
        data = self.model_output_raw(response)
        # Try to parse the input JSON string and complete it if it is incomplete
        data["parsed"] = Parser().fit(data["text"], json_depth_limit)
        return data
    
    def model_output_raw(self, response: ChatCompletion) -> Dict:
        data = {}
        status_code = response.choices[0].finish_reason
        assert status_code == "stop", f"The status code was {status_code}."
        content = response.choices[0].message.content
        data["text"] = content.strip()
        # data["usage"] = dict(response.usage)
        return data

    def _initialize_encoder(self):
        self.encoder = tiktoken.encoding_for_model(self.model_name)
