import numpy as np

class WCSTSCORER:

    def __init__(self, sessions):
        self.sessions = sessions
    
    def scoringwcst(self, results):
        wcst_scores = []
        for session in range(self.sessions):
            wcst_score = []
            for i in range(len(results[session])):
                wcst_score.append(results[session][i]['feedback'])
            wcst_scores.append(np.sum(wcst_score) /  len(results[session]))
        score = np.mean(wcst_scores) * 100
        return score
    