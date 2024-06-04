import tenacity
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Union, Mapping, Iterator, Any


class BaseModel(metaclass=ABCMeta):
    """
    Abstract base class for a large language models (LLMs).

    Args:
        api_key (str): API key to identify an application.
        model_name (str): Name of the model.
        api_await (bool): Waiting time for the API to finish in seconds.
        api_retry (int): Retrying time for the API to finish.
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

        Args:
            api_key (str): API key to identify an application.
        """
        raise NotImplementedError
    
    @abstractmethod
    def supported_models(self) -> List[str]:
        """
        Get a list of supported models for the endpoint.

        Returns:
            List of supported models.
        """
        raise NotImplementedError
    
    @abstractmethod
    def set_model_name(self, model_name: str):
        """
        Set model name for the endpoint.

        Args:
            model_name (str): Name of the model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get model description.

        Returns:
            Return the description of the model class.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_endpoint(self) -> str:
        """
        Get model endpoint.
        
        Returns:
            Return the endpoint of the model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, str]:
        """
        Get model parameters.

        Returns:
            Get the parameters of the model.
        """
        raise NotImplementedError
    
    @abstractmethod
    def run(self, prompt: str) -> Union[Mapping[str, Any], Iterator[Mapping[str, Any]]]:
        """
        Run the LLM on the given prompt.

        Args:
            prompt (str): It serves as a form of conditioning that guides the model's output.
        """
        raise NotImplementedError
    
    @abstractmethod
    def model_output(self, response: Any) -> Dict:
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