import tenacity
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Union, Optional


class BaseModel(metaclass=ABCMeta):
    """
    Abstract base class for a large language models (LLMs).
    """
    name = ""
    description = ""

    def __init__(
        self, 
        api_key: str, 
        model_name: str, 
        api_await: int = 60, 
        api_retry: int = 5, 
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.api_await = api_await
        self.api_retry = api_retry
        self._verify_model()
        self.set_key(api_key)

    @abstractmethod
    def _verify_model(self):
        """
        Verify the model is supported by the endpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def set_key(self, api_key: str):
        """
        Set endpoint API key if needed.
        """
        raise NotImplementedError
    
    @abstractmethod
    def supported_models(self) -> List[str]:
        """
        Get a list of supported models for the endpoint.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_model_name(self, model_name: str):
        """
        Set model name for the endpoint.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get model description.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_endpoint(self) -> str:
        """
        Get model endpoint.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, str]:
        """
        Get model parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def run(self, prompts: List[str]) -> List[str]:
        """
        Run the LLM on the given prompt list.
        """
        raise NotImplementedError
    
    @abstractmethod
    def model_output(self, response):
        """
        Get the model output from the response.
        """
        raise NotImplementedError
    
    def _retry_decorator(self):
        """
        Decorator function for retrying API requests if they fail.
        """
        return tenacity.retry(
            wait=tenacity.wait_random_exponential(
                multiplier=0.3, exp_base=3, max=self.api_await
            ),
            stop=tenacity.stop_after_attempt(self.api_retry),
        )

    def execute_with_retry(self, *args, **kwargs):
        """
        Decorated version of the run method with the retry logic.
        """
        decorated_run = self._retry_decorator()(self.run)
        return decorated_run(*args, **kwargs)