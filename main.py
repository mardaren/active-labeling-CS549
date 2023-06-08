from config.config_reader import config_dict
from data_loader import DataLoader
from models.active_labeler import ActiveLabeler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
def main():

    data_loader = DataLoader(config_dict=config_dict)
    # Binary
    # model = SVC(C=0.1, kernel="poly", decision_function_shape='ovo', probability=True)
    # model = SVC(C=3, kernel="rbf", decision_function_shape='ovr', probability=True)
    #
    # X_train, X_test, y_train, y_test = train_test_split(data_loader.embeddings, data_loader.labels, test_size=0.9, random_state=28)
    #
    # model.fit(X_train, y_train)
    # result = model.predict(X_test)
    #
    # f1 = f1_score(y_true=y_test, y_pred=result, average='macro')
    # print(f1)

    # model.fit(data_loader.embeddings, data_loader.labels)
    # result = model.predict(data_loader.embeddings)
    #
    # f1 = f1_score(y_true=data_loader.labels, y_pred=result)
    # print(f1)

    # return
    active_labeler = ActiveLabeler(config_dict=config_dict, data_x=data_loader.embeddings,
                                   data_y=data_loader.labels)
    active_labeler.run()
    print(data_loader.embeddings)


if __name__ == "__main__":
    main()
