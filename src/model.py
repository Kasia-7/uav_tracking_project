from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_type="random_forest"):
        if model_type == "random_forest":
            return RandomForestClassifier()
        else:
            raise ValueError("不支持的模型类型")

class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
