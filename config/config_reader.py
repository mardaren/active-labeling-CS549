import yaml

available_datasets = ["imdb", "yelp-review", "tweet-eval", "tweet-topic"]
available_models = ["classifier-unc", "edl-classification"]
# imdb -> binary
# yelp-review -> multiclass-formal
# tweet-eval -> multiclass-informal
# tweet-topic -> multilabel

config_dict = None
with open("config/config.yml", "r") as stream:
    try:
        config_dict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
