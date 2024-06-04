import re
from textwrap import dedent

from ollama_prompter import Prompter, Pipeline, Ollama


def main():
    # Define text input and pre-defined labels
    text = dedent("""
        The Federal Bureau of Investigation has been ordered to track down as many as 
        3000 Iraqis in this country whose VISAs have expired, the Justice Department 
        said yesterday.
    """)
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    labels = ['ORGANIZATION', 'LOCATION', 'DATE']

    # Define model, prompter and pipeline
    model = Ollama(
        model_name='llama3:latest', 
        endpoint='http://localhost:11434', 
        temperature=0.1, 
        top_k=1, 
        top_p=1
    )
    prompter = Prompter(
        template_name='ner.jinja', 
        template_dir='templates'
    )
    pipe = Pipeline([prompter] , model, json_depth_limit=50)

    # Inference
    variables = {
        'labels': labels, 
    }
    result = pipe.fit(
        text=text, 
        verbose=True, 
        **variables
    )
    # [{'T': 'Organization', 'E': 'Federal Bureau of Investigation'}, {'T': 'Location', 'E': 'Iraqis'}, {'T': 'Organization', 'E': 'Justice Department'}]
    print(eval(result[0]['text']))


if __name__ == '__main__':
    main()