class METAPRTSCORER:

    def __init__(self, result, sessions, trials):
        self.result = result
        self.sessions = sessions
        self.trials = trials


    def scoring_metaprt(self):
        scores = []
        for session in range(self.sessions):
            score = 0
            for i in range(self.trials):
                if self.result[session][i]['Reverse'] == True and (i+1)<self.trials:
                    if 0 in [self.result[session][i]['reward'],self.result[session][i-1]['reward'],self.result[session][i+1]['reward']]:
                        score += 0
                    else:
                        score += 1
                else:
                    pass
            scores.append(score)
    
        return scores
