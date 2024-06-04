from ollama_prompter import Prompter, Pipeline, Ollama


def main():
    # Define text input and pre-defined labels
    text = "Cikamatana reaches the end of long road to the Olympics."
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Define model, prompter and pipeline
    model = Ollama(
        model_name='phi3:medium', # llama3:latest
        endpoint='https://octopus-ideal-piglet.ngrok-free.app', 
        temperature=0.1, 
        top_k=1, 
        top_p=1
    )
    prompter = Prompter(
        template_name='text_classification.jinja', 
        template_dir='templates'
    )
    pipe = Pipeline([prompter] , model, json_depth_limit=100)

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


if __name__ == '__main__':
    main()