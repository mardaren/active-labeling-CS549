from config.config_reader import config_dict
from data_loader import DataLoader
from models.active_labeler import ActiveLabeler

def main():

    data_loader = DataLoader(config_dict=config_dict)
    active_labeler = ActiveLabeler(config_dict=config_dict, data_x=data_loader.embeddings,
                                   data_y=data_loader.labels)
    active_labeler.run()
    print(data_loader.embeddings)


if __name__ == "__main__":
    main()
