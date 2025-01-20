import numpy as np
import time
from utils.extractor import EXTRACTOR
from utils.subject import Subject
from tqdm import tqdm
from scorer.metaprt_scorer import METAPRTSCORER

class MetaPRT:

    def __init__(self, key_subject, url_subject, model_name, config, COT=None):

        self.extractor = EXTRACTOR(task="meta probability reversal task_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="meta probability reversal task", model=model_name)
        self.p = config['metaprt']['p']
        self.sessions = config['metaprt']['sessions']
        self.trials = config['metaprt']['trials']
        self.interval_1 = config['metaprt']['interval_1']
        self.interval_2 = config['metaprt']['interval_2']
        self.model = model_name
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis. "


    def instruction(self, trail_no, prior_reward):

        instruction = "This is the first trial, choose one arm please." if trail_no == 1 else f"Your previous choice resulted in a reward of {prior_reward}. This is trial number {trail_no}. Please choose the option you think is best. "

        return instruction + self.cot_instruction

    def get_reward(self, side, prob):
        q = 1 - prob
        if side == 'AAA':
            reward = np.random.binomial(n=1, p=prob, size=1)[0] # AAA for right, therefore p is the probability of getting reward of right arm
        elif side == 'BBB':
            reward = np.random.binomial(n=1, p= q, size=1)[0] # BBB for left
        
        return reward

    def metaprt(self, trials, interval_1, interval_2, pbar=None):

        conversation = None
        outcomes = {}
        reward = 0
        half_trials = trials // 2
        prob = self.p
        
        for i in range(trials):
            try:

                if i < half_trials:
                    interval = interval_1
                    mark = i
                else:
                    interval = interval_2
                    mark = i - half_trials

                group = mark // interval

            
                if i>0 and mark%interval == 0: 
                    prob= 1-prob
                    reverse = True
                else:
                    prob = prob
                    reverse = False

                trial_no = i + 1
                instruction = self.instruction(trial_no, reward)
                reply, conversation = self.subject.conversation(instruction, conversation)
                decision = self.extractor.extraction(reply)
                reward = self.get_reward(decision, prob)
                outcomes[i] = {'model reply': reply, 'decision': decision, 'reward': reward, 'Reverse': reverse}

            except (RuntimeError, ValueError) as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'decision': 'ERROR', 'reward': None}
            
            if pbar:
                pbar.update(1)
            time.sleep(0)
            #print(outcomes)

        return outcomes, conversation


    def test_metaprt(self):
        all_sessions_outcomes = {}
        conversations = {}
        total_trials = self.trials * self.sessions

        with tqdm (total=total_trials, desc='Testing Meta-Multi Bandit Task') as pbar:
            for session in range(self.sessions):
                outcomes, conversation = self.metaprt(self.trials, self.interval_1, self.interval_2, pbar)
                all_sessions_outcomes[session] = outcomes
                conversations[session] = conversation
        return all_sessions_outcomes, conversations
    
    def evaluate_metaprt(self):
        results_metaprt, conversations_metaprt = self.test_metaprt()
        try:
            metaprt_scorer = METAPRTSCORER(results_metaprt, self.sessions, self.trials)
            score_metaprt = metaprt_scorer.scoring_metaprt()
            print(f"Score of Meta-Multi Bandit Task of {self.model}: ", score_metaprt)
                    
        except Exception as e:
            print(f"Error occurred: {str(e)}.")
            score_metaprt = None
        
        return results_metaprt, conversations_metaprt, score_metaprt

