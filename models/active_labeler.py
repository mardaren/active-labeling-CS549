import numpy as np
from numpy.random import default_rng

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
from sklearn.svm import SVC

from sklearn.metrics import f1_score

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier

class ActiveLabeler:

    def __init__(self, config_dict: dict, data_x: np.array, data_y: np.array, test: bool = True, limited: bool=True):
        self.config_dict = config_dict
        self.al_model = None
        self.threshold = 0.01
        self.limited = limited

        self.data_x = data_x
        if test:
            self.data_y = data_y

        if config_dict["dataset"] == 'imdb':
            self.average = 'binary'
        elif config_dict["dataset"] == 'tweet-topic':
            self.average = 'weighted'
        else:
            self.average = 'macro'

        self.get_al_model()

    def get_al_model(self):
        model_name = self.config_dict["model"]
        if model_name == "classifier-unc":
            if self.config_dict['dataset'] == 'tweet-topic':
                estimator = MLPClassifier(max_iter=500)
            else:
                estimator = SVC(C=0.1, kernel="poly", decision_function_shape='ovo', probability=True)
            self.al_model = ActiveLearner(estimator=estimator, query_strategy=entropy_sampling)

    def run(self):
        model_name = self.config_dict["model"]
        if model_name == "classifier-unc":
            self.run_al()

    def teach(self, x_train, y_train):
        if self.config_dict["model"] == "classifier-unc":
            self.al_model.teach(x_train, y_train)

    def get_uncertainty_metric(self, y_pred):
        if self.config_dict["model"] == "classifier-unc":
            if self.config_dict["dataset"] == "tweet-topic":
                return np.mean(np.abs((y_pred > 0.5) - y_pred), axis=1)
            else:
                return 1 - np.max(y_pred, axis=1)

    def run_al(self):
        percentage_samples = [0]
        percentage_auto = [0]
        f1_s = [0]
        time = [0]

        human_labeled = 100
        auto_labeled = 0
        # create model-labels as sparse array
        auto_label_idx = []

        model_labels = np.full(self.data_y.shape, -1)

        rng = default_rng()
        known_idx = np.array(rng.choice(len(self.data_x), size=human_labeled, replace=False))
        unknown_idx = np.array(list(set(range(len(self.data_x))) - set(known_idx)))

        x_start = self.data_x[known_idx]
        y_start = self.data_y[known_idx]

        self.teach(x_start, y_start)
        model_labels[known_idx] = y_start
        i = 0

        while True:
            # PART I: Auto-labeling phase
            y_pred = self.al_model.estimator.predict_proba(self.data_x[unknown_idx])
            uncertainty_metrics = self.get_uncertainty_metric(y_pred)

            cand_rel_idx = np.where(uncertainty_metrics < self.threshold)[0]  # these are index of index !!!

            if len(cand_rel_idx) > 0:
                # Get the labels
                cand_y_pred = y_pred[cand_rel_idx]
                cand_idx = unknown_idx[cand_rel_idx]
                if self.config_dict["dataset"] == "tweet-topic":
                    model_labels[cand_idx] = cand_y_pred > 0.5
                else:
                    model_labels[cand_idx] = np.argmax(cand_y_pred, axis=1)
                auto_labeled += len(cand_idx)

                # Discard cand_rel_idx from uncertainty metrics
                uncertainty_metrics = np.delete(uncertainty_metrics, cand_rel_idx, axis=0)

                # Change known and unknown indices
                known_idx = np.concatenate((known_idx, cand_idx), axis=0)
                unknown_idx = np.delete(unknown_idx, cand_rel_idx, axis=0)

                if len(unknown_idx) == 0:
                    print("*************************************")
                    print(f"Human labeled: {human_labeled}")
                    print(f"Auto labeled: {auto_labeled}")
                    print(f"Unlabeled: {len(unknown_idx)}")
                    f1 = f1_score(y_true=self.data_y[known_idx], y_pred=model_labels[known_idx], average=self.average)
                    print(f"f1 score: {f1}")
                    return

            # PART II: Querying phase
            # query_result = self.al_model.query(self.data_x[unknown_idx])
            rel_query_idx = np.argmax(uncertainty_metrics)
            query_idx = unknown_idx[rel_query_idx]

            # Get the label
            # self.al_model.teach(self.data_x[sample_idx].reshape(1, -1), [self.data_y[sample_idx]])
            self.teach(self.data_x[query_idx].reshape(1, -1), [self.data_y[query_idx]])

            # Add the label
            model_labels[query_idx] = self.data_y[query_idx]
            human_labeled += 1

            # Change known and unknown indices
            known_idx = np.append(known_idx, query_idx)
            unknown_idx = np.delete(unknown_idx, rel_query_idx, axis=0)

            percentage_samples.append((human_labeled + auto_labeled) / len(self.data_y))
            percentage_auto.append(auto_labeled / human_labeled)
            f1_s.append(f1_score(y_true=self.data_y[known_idx], y_pred=model_labels[known_idx], average=self.average))
            time.append(i)

            if i % 10 == 0 or len(unknown_idx) == 0:
                print("*************************************")
                print(f"Human labeled: {human_labeled}")
                print(f"Auto labeled: {auto_labeled}")
                print(f"Unlabeled: {len(unknown_idx)}")
                f1 = f1_s[-1]
                print(f"f1 score: {f1}")

                if self.limited and i >= 300:
                    plt.plot(time, percentage_samples, label='labeled/total')
                    plt.plot(time, percentage_auto, label='auto/human')
                    plt.plot(time, f1_s, label='f1 score')
                    plt.legend(loc="upper left")
                    plt.savefig('binary.png')
                    plt.show()
                    return

                if len(unknown_idx) == 0:
                    return
            i += 1
