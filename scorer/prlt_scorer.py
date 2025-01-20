import numpy as np

class PRLTSCORER:

    def __init__(self,results, sessions, trials, p, windowsize):
        self.results = results
        self.sessions = sessions
        self.trials = trials
        self.p = p
        self.windowsize = windowsize

    def overall_rewards(self, result, sessions, trials):
        scores = []
        for session in range(sessions):
            reward = 0
            for trial in range(trials):
                reward += result[session][trial]['reward']
        
            score = (reward / trials) * 100

            scores.append(score)

        return scores

    
    def true_distribution(self, trials, p):
        half_trials = int(trials / 2)
        return np.concatenate([np.ones(half_trials) * p, np.ones(half_trials) * (1-p)])

    def meanerror(self, est, true, p):
        errs = []
        for i in range(len(est)):
            err = np.abs(est[i]-true[i])
            errs.append(err)

        all = np.sum(errs)
        max = p * len(est)
        mean = 1 - (all/max) 
        return mean

    def estimatingprior(self, decisions, windowsize):

        alpha_prior = 0.1
        beta_prior = 0.1
        theta_estimates = []
        for t in range(len(decisions)):
            start = max(0, t - windowsize + 1)
            window = decisions[start:t + 1]
            n_1 = sum(window)
            n_0 = len(window) - n_1
            theta = (alpha_prior + n_1) / (alpha_prior + n_1 + beta_prior + n_0)
            theta_estimates.append(theta)

        return theta_estimates

    def scoringprlt(self):

        true_dist = self.true_distribution(self.trials, self.p)
        target_decision = 'AAA'
        scores = []
        reward_scores = self.overall_rewards(self.results, self.sessions, self.trials)

        for session in range(self.sessions):
        
            decisions = []
            for trial in range(self.trials):
                decision = 1 if self.results[session][trial]['extraction'] == target_decision else 0
                decisions.append(decision)
        
            priors_score = self.meanerror(self.estimatingprior(decisions,self.windowsize), self.estimatingprior(true_dist,self.windowsize),self.p) * 100
            reward_score = reward_scores[session]
            scores.append(np.mean([priors_score, reward_score]))
    
        return np.mean(scores)        