# Ollama Prompter

This repository is dedicated to enhancing the functionality originally found in the [Promptify](https://github.com/promptslab/Promptify) GitHub repo. As the original Promptify repository is no longer actively maintained, we have decided to adapt and extend its capabilities by integrating the Ollama API to facilitate prompt engineering to solve NLP problems.

# Installation

```bash
pip install -e .
```

# Quick Tour

```python
from ollama_prompter import Ollama, Prompter, Pipeline

text_input = "Wall Street's dwindling and of ultra-cynics, are seeing green again."
labels = {'World', 'Sports', 'Business', 'Sci/Tech'}
model = Ollama(
    model_name='llama3:latest', 
    endpoint='http://localhost:11434', 
    temperature=0.1, 
    top_k=1, 
    top_p=1
)
prompter = Prompter('multiclass_classification.jinja')
pipe = Pipeline([prompter] , model)

result = pipe.fit(
    text_input=text_input, 
    labels=labels, 
    description=description, 
)
print(eval(result[0]['text']))
```

# Features

 -  Ollama API integration to enhance the prompt engineering capability of the [Promptify](https://github.com/promptslab/Promptify) origninal code.

# Acknowledgements

We have utilised a substantial amount of code from the [Promptify](https://github.com/promptslab/Promptify) GitHub repository. We are grateful to the original authors for their groundwork, which has been instrumental in the progress of this project.