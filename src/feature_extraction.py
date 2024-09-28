class FeatureExtractor:
    def __init__(self, data):
        self.data = data
    
    def extract_basic_features(self):
        return self.data[['斜距(m)', '方位角（°）', '俯仰角（°）']]

    def extract_velocity_features(self):
        return self.data['径向速度（m/s）']
