import numpy as np

class DCIGTSCORER:

    def __init__(self, sessions, trials, result):
        self.sessions = sessions
        self.trials = trials
        self.result = result


    def beneficial_switching(self):

        counts = []
        scores = []

        for session in range(len(self.result)):
            count = 0
            score = 0

            for i in range(len(self.result[session])):
                try:

                    if self.result[session][i]['selection 1'] == self.result[session][i]['selection 2']:
                        # insisting for expected gain
                        if self.result[session][i]['loss 1'] == 0:
                            score += 1
                        # insisting without avoiding risks
                        elif self.result[session][i]['loss 1'] > 0:
                            score -= (self.result[session][i]['loss 1'] / 500) + 1
            
                    elif self.result[session][i]['selection 1'] != self.result[session][i]['selection 2']:
                        if self.result[session][i]['loss 1'] == 0:
                            # switching unnecessarily
                            score -= 0.5
                        elif self.result[session][i]['loss 1'] > 0:            
                            # switching for less loss
                            score += 1
                
                    count += 1
                except Exception as e:
                    print(e, f'skip scoring trial {i}')


            scores.append(score)
            counts.append(count)

        return scores,counts

    def scoring_dcgit(self):
        beneficialswitches_scores, beneficialswitches_counts = self.beneficial_switching()
        overall_scores = []
        pure_scores = []

        for session in range(self.sessions):
        
            switch_score = (beneficialswitches_scores[session] / beneficialswitches_counts[session]) * 100
            pure_scores.append(switch_score)

            final_overage = self.result[session][self.trials - 1]['overage']
            max_overage = 2000 + (100 * self.trials)
            overage_score = (final_overage / max_overage) * 100

            overall_score = (switch_score + overage_score) / 2

            overall_scores.append(overall_score)

        return [np.mean(overall_scores), np.mean(pure_scores)]
