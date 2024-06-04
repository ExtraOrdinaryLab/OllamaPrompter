from ollama_prompter import Prompter, Pipeline, Ollama


def main():
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
    pipe = Pipeline([prompter] , model, json_depth_limit=100)

    # Inference
    variables = {
        'labels': labels, 
        'chain_of_thought': True, 
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
                'text': 'India vote count shows Modi alliance heading to majority but no landslide.', 
                'reason': 'The text discusses political elections in India, a significant international event.', 
                'answer': 'World'
            }, 
            {
                'text': 'British scientists said on Wednesday they had received permission to clone human embryos for medical research.', 
                'reason': 'The text discusses scientific activity related to cloning and medical research, highlighting a technological and ethical development.', 
                'answer': 'Sci/Tech'
            }, 
            {
                'text': 'Cristiano Ronaldo makes Manchester United transfer request after breaking down in tears.', 
                'reason': 'The text mentions Cristiano Ronaldo and Manchester United, which are related to soccer, a sports topic.', 
                'answer': 'Sports'
            }, 
            {
                'text': "Bitcoin Knocks on $70K Level; Bitfinex Hopeful Selling Pressure That Sparked a Correction Is Ending.", 
                'reason': "The text focuses on Bitcoin's price level and market behavior, which are topics directly related to financial markets.", 
                'answer': 'Business'
            }
        ]
    }
    result = pipe.fit(
        text=text, 
        verbose=False, 
        **variables
    )
    print(eval(result[0]['text'])) # [{'C': 'Sports', 'R': '...'}]


if __name__ == '__main__':
    main()