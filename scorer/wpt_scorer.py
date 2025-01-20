from ast import literal_eval
import numpy as np

class WPTSCORER:

    def __init__(self, sessions, result, estimate_interval):
        self.sessions = sessions
        self.result = result
        self.estimate_interval = estimate_interval
    
    def estimate_transition(self):
        weather_map = {'sunny': 0, 'rainy': 1}
        estimated_transitions = []
        true_transitions = []
    
        for session in range(self.sessions):
            pred_transitions = [np.zeros((2, 2)) for _ in range(2)]
            pred_counts = [np.zeros((2, 2)) for _ in range(2)]
            real_transitions = [np.zeros((2, 2)) for _ in range(2)]
            real_counts = [np.zeros((2, 2)) for _ in range(2)]
            estimated_interval = {i: v for i, v in enumerate([self.result[session][k] for k in range(self.estimate_interval, len(self.result[session]))])}
        
            for i in range(len(estimated_interval)):
                today = weather_map[estimated_interval[i]['today weather']]
                device = literal_eval(estimated_interval[i]['device state'])
                prediction = weather_map[estimated_interval[i]['model prediction'].lower()]
                tomorrow = weather_map[estimated_interval[i]['tomorrow weather']]
                active_device = 0 if device[0] == 1 else 1
                pred_counts[active_device][today][prediction] += 1
                real_counts[active_device][today][tomorrow] += 1
            
            for dev in range(2): # estimating model's transition probabilities

                for i in range(2):
                    row_sum = np.sum(pred_counts[dev][i])
                    if row_sum > 0:
                        pred_transitions[dev][i] = pred_counts[dev][i] / row_sum
                    
            for dev in range(2): #estimating real transition probabilities
                for i in range(2):
                    row_sum = np.sum(real_counts[dev][i])
                    if row_sum > 0:
                        real_transitions[dev][i] = real_counts[dev][i] / row_sum
                    
            estimated_transitions.append(pred_transitions)
            true_transitions.append(real_transitions)
    
        return estimated_transitions, true_transitions

    def transition_diff(self, est_trans, true_trans):
        assert len(est_trans) == len(true_trans)    
        differences = []
        for i in range(len(est_trans)):
            diff_pair = []
            for j in range(len(est_trans[i])):
                diff = np.linalg.norm(est_trans[i][j] - true_trans[i][j], 'fro')
                diff_pair.append(diff)
            differences.append(diff_pair)        
        return differences

    def max_diffs(self, true_transition):
        for k, transition in enumerate(true_transition):
            matrix = np.array(transition)
            max_diff = 0
            for i in range(2):
                for j in range(2):
                    error_0 = matrix[i, j]
                    error_1 = 1 - matrix[i, j]
                    max_diff += max(error_0, error_1)
        return max_diff


    def scoring_wpt(self):
        estimated_transitions, true_transitions = self.estimate_transition()
        diffs = self.transition_diff(estimated_transitions, true_transitions)
        scores = []
        for session in range(self.sessions):
            diff = np.sum(diffs[session])
            max_diff = self.max_diffs(true_transitions[session])
            score = (1 - diff/max_diff) * 100
            scores.append(score)
        return np.mean(scores)
    