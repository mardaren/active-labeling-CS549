import yaml

data = {
    "dataset": "imdb",
    "model": "bayesian-opt",
    "tokenizer": "bert-base-uncased"
}

with open('config.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
