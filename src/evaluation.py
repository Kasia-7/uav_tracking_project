from sklearn.metrics import accuracy_score

class Evaluation:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Evaluation, cls).__new__(cls)
        return cls._instance

    def calculate_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
