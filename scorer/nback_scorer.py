import numpy as np

class NBACK_SCORER:
    
    def __init__(self, result, sessions, length):
        self.result = result
        self.sessions = sessions
        self.length = length
    
    def scoringnback(self):
        corrects = []
        for i in range(self.sessions):
            correct = np.sum([item['is_correct'] for item in self.result[i].values()])
            corrects.append(correct)        
        mean_acc = np.sum(corrects) / (self.length * self.sessions) * 100

        return mean_acc
