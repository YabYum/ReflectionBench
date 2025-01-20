from utils.extractor import EXTRACTOR
from utils.subject import Subject
import numpy as np
import time
from tqdm import tqdm
from scorer.prlt_scorer import PRLTSCORER

class PRT:

    def __init__(self, key_subject, url_subject, model_name, config, COT=None):
        self.extractor = EXTRACTOR(task="probability reversal task_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="probability reversal task", model=model_name)
        self.p = config['prlt']['p']
        self.sessions = config['prlt']['sessions']
        self.trials = config['prlt']['trials']
        self.model = model_name
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis."

    def instruction(self, trail_no, prior_reward):

        instruction = "This is the first trial, choose one arm please. " if trail_no == 1 else f"Your previous choice resulted in a reward of {prior_reward}. This is trial number {trail_no}. Please choose the option you think is best. "

        return instruction + self.cot_instruction
    
    def reward(self, side, prob):

        return np.random.binomial(n=1, p=prob if side == 'AAA' else 1-prob, size=1)[0]

    def prt(self, trials, pbar=None):
        conversation = None
        outcomes = {}
        reward = 0
        rp = 1 - self.p
        
        
        for i in range(trials):
            trial_no = i + 1
            instruction = self.instruction(trial_no, reward)
            try:
                reply, conversation = self.subject.conversation(instruction, conversation)
                decision = self.extractor.extraction(reply)
                reward = self.reward(decision, self.p if i < trials/2 else rp)
                outcomes[i] = {'reply': reply, 'extraction': decision, 'reward': reward}

            except Exception as e:

                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'reply': None, 'extraction': None, 'reward': None}
        
            if pbar:
                pbar.update(1)    
            time.sleep(0)       

        return outcomes, conversation

    def test_prt(self, trials, sessions):
        all_sessions_outcomes = {}
        all_conversations = {}
        total_trials = trials * sessions
        pbar = None

        with tqdm (total=total_trials, desc='Testing Probabilistic Reversal Learning Task: ') as pbar:
            for session in range(sessions):
                
                outcomes, conversation = self.prt(trials, pbar)            
                all_sessions_outcomes[session] = outcomes
                all_conversations[session] = conversation
        
        return all_sessions_outcomes, all_conversations
    
    def evaluate_prlt(self):
        results_prlt, conversations_prlt = self.test_prt(self.trials, self.sessions)
        try:
            scorer_prlt = PRLTSCORER(results_prlt, self.sessions, self.trials, self.p, windowsize=5)
            score_prlt = scorer_prlt.scoringprlt()
            print(f"Score of Probalistic Reversal Learning Task of {self.model}: ", score_prlt)
        
        except Exception as e:
            print("Error: ", e)
            score_prlt = None

        return results_prlt, conversations_prlt, score_prlt
    