# OllamaPrompter

This repository is dedicated to enhancing the functionality originally found in the [Promptify](https://github.com/promptslab/Promptify) GitHub repo. As the original Promptify repository is no longer actively maintained, we have decided to adapt and extend its capabilities by integrating the Ollama API to facilitate prompt engineering to solve NLP problems.

# ⚙️ Installation

This repository is tested on Python 3.9+, macOS Sonoma and Ubuntu 22.04.

```bash
git clone https://github.com/penguinwang96825/OllamaPrompter.git
pip install -e .
```

# ⛩️ Quick Tour

```python
from ollama_prompter import Ollama, Prompter, Pipeline

# Define text input and pre-defined labels
text = "Cikamatana reaches the end of long road to the Olympics."
labels = ['World', 'Sports', 'Business', 'Sci/Tech']

# Define model, prompter and pipeline
model = Ollama(
    model_name='llama3:latest', 
    endpoint='http://localhost:11434', 
    temperature=0.1, 
    top_k=1, 
    top_p=1
)
prompter = Prompter(
    template_name='text_classification.jinja', 
    template_dir='templates'
)
pipe = Pipeline([prompter] , model, json_depth_limit=20)

# Inference
variables = {
    'labels': labels, 
    'exclusive_classes': True, 
}
result = pipe.fit(
    text=text, 
    verbose=False, 
    **variables
)
print(eval(result[0]['text'])) # [{'C': 'Sports'}]
```

More examples will be forthcoming in [examples](https://github.com/penguinwang96825/OllamaPrompter/tree/master/examples) folder.

# 🎮 Features

 - Ollama API integration to enhance the prompt engineering capability of the [Promptify](https://github.com/promptslab/Promptify) origninal code.
 - Chain-of-Thought (CoT) is supported in text classification task to encourage LLMs to explain their reasonings.

# 💡 Roadmaps

 - [ ] Add local HuggingFace models support to work with current OllamaPrompter.
 - [x] Chain-of-Thought (CoT) integration in text classification task to encourage LLMs to explain their reasonings.
 - [ ] Self-Consistency (SC) integration to send the same prompt with the same text to the same LLM multiple times with different reasoning paths.
 - [ ] Perplexity Estimation of the prompt to measure LLMs awareness and confidence.
 - [ ] OpenAI API tokens costs estimation.
 - [ ] Add more NLP tasks.
     - [x] Text Classification
     - [ ] Named Entity Recognition (NER)
     - [ ] Question Answering (QA)
     - [ ] Summarisation
     - [ ] Translation

# License

This project is licensed under the [MIT License](https://github.com/penguinwang96825/OllamaPrompter/blob/main/LICENSE).

# Credits

Inspired by and based on code from [Promptify](https://github.com/promptslab/Promptify) and [Spacy-LLM](https://github.com/explosion/spacy-llm). We have utilised a substantial amount of code from these repos, and we are grateful to the original authors for their groundwork, which has been instrumental in the progress of this project.

# Contact

Please reach out to [yangwang4work@gmail.com](mailto:yangwang4work@gmail.com) for any questions related to this code. If you find this repo useful, we would also be happy about a ⭐️.

# 📝 Acknowledgements

 - Thanks to everyone who contribute their time, ideas and codes to OllamaPrompter.
 - Thanks to the excelent projects: [Promptify](https://github.com/promptslab/Promptify), [Spacy-LLM](https://github.com/explosion/spacy-llm), [Ollama][https://github.com/ollama/ollama-python], [OpenAI](https://github.com/openai/openai-python), [Transformers](https://github.com/huggingface/transformers), [Jinja](https://github.com/pallets/jinja), etc.