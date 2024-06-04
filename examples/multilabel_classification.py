import re
from textwrap import dedent

from ollama_prompter import Prompter, Pipeline, Ollama


def main():
    # Define text input and pre-defined labels
    text = dedent("""
        Large deep neural networks are powerful, but exhibit undesirable behaviors
        such as memorization and sensitivity to adversarial examples. In this work, we
        propose mixup, a simple learning principle to alleviate these issues. In
        essence, mixup trains a neural network on convex combinations of pairs of
        examples and their labels. By doing so, mixup regularizes the neural network to
        favor simple linear behavior in-between training examples. Our experiments on
        the ImageNet-2012, CIFAR-10, CIFAR-100, Google commands and UCI datasets show
        that mixup improves the generalization of state-of-the-art neural network
        architectures. We also find that mixup reduces the memorization of corrupt
        labels, increases the robustness to adversarial examples, and stabilizes the
        training of generative adversarial networks.
    """)
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    labels = [
        'Computer Science', 'Physics' , 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'
    ]

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
    pipe = Pipeline([prompter] , model, json_depth_limit=50)

    # Inference
    variables = {
        'labels': labels, 
        'num_labels': len(labels), 
        'exclusive_classes': False, 
        'allow_none': False, 
    }
    result = pipe.fit(
        text=text, 
        verbose=True, 
        **variables
    )
    print(eval(result[0]['text'])) # [{'C': 'Computer Science'}, {'C': 'Statistics'}]


if __name__ == '__main__':
    main()