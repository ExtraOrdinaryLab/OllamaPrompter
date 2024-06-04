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
text = "Wall Street's dwindling and of ultra-cynics, are seeing green again."
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
pipe = Pipeline([prompter] , model)

# Inference
variables = {
    'labels': labels, 
    'exclusive_classes': True, 
    'allow_none': False, 
    'label_definitions': {
        'World': 'This category typically covers international news, events, and issues.', 
        'Sports': 'News in this category focuses on athletic competitions, events, and achievements.', 
        'Business': 'This label includes news related to the economic sector and commerce.', 
        'Sci/Tech': 'This category encompasses news related to scientific discoveries and technological advancements.', 
    }, 
    'prompt_examples': [
        {
            'text': 'British scientists said on Wednesday they had received permission to clone human embryos for medical research', 
            'answer': 'Sci/Tech'
        }
    ]
}
result = pipe.fit(
    text=text, 
    **variables
)
print(eval(result[0]['text'])) # ["C": "Business"]
```

# 🎮 Features

 -  Ollama API integration to enhance the prompt engineering capability of the [Promptify](https://github.com/promptslab/Promptify) origninal code.

# 📝 Acknowledgements

We have utilised a substantial amount of code from the [Promptify](https://github.com/promptslab/Promptify) GitHub repository. We are grateful to the original authors for their groundwork, which has been instrumental in the progress of this project.