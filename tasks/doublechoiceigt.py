import numpy as np
import time
from utils.extractor import EXTRACTOR
from utils.subject import Subject
from tqdm import tqdm
from scorer.dcigt_scorer import DCIGTSCORER


class DoubleChoiceIGT:

    def __init__(self, key_subject, url_subject, model_name, config, COT=None):
         
        self.extractor = EXTRACTOR(task="double choice iowa gambling test_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="double choice iowa gambling test", model=model_name)
        self.sessions = config['dcigt']['sessions']
        self.trials = config['dcigt']['trials']
        self.p_loss = config['dcigt']['p_loss']
        self.num_loss = config['dcigt']['num_loss']
        self.num_gain = config['dcigt']['num_gain']
        self.model = model_name     
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis"
    
    def instruction_1(self,trial_no,overage,gain=None,loss=None): # instructing the first choice
        if gain is None: 
            instruction = f"The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        else:
            instruction = f"The outcome of your last choice is gain: ${gain}, loss: ${loss}. The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        return instruction + self.cot_instruction
    
    def instruction_2(self,overage,gain=None,loss=None): # instructing the second choice

        instruction = f"If your choose this deck, the outcome will be gain: ${gain}, loss: ${loss}. the money in your account will ${overage}. Which deck would you choose? Make your final choice!"

        return instruction + self.cot_instruction
    
    def outcome(self,choice):
        p_loss = self.p_loss[choice]
        num_loss = self.num_loss[choice]
        loss =  np.random.binomial(n=1, p=p_loss, size=1)[0] * num_loss
        gain = self.num_gain[choice]
        return loss, gain
    
    def dcigt(self, pbar=None):
        conversation = None
        gain = None
        loss = None
        account = 2000
        outcomes = {}

        for i in range(self.trials):
            try:
                trial_no = i+1
                instruction_1 = self.instruction_1(trial_no, account, gain, loss)

                reply_1, conversation = self.subject.conversation(instruction_1, conversation)
                selection_1 = self.extractor.extraction(reply_1)
                loss_1, gain_1 = self.outcome(selection_1)
                account_1 = account + gain_1 - loss_1
                instruction_2 = self.instruction_2(account_1,gain_1,loss_1)
                reply_2, conversation = self.subject.conversation(instruction_2, conversation)
                selection_2 = self.extractor.extraction(reply_2)
                loss, gain = self.outcome(selection_2)
                account += (gain - loss)

                outcomes[i] = {'model reply 1': reply_1, 'selection 1': selection_1, 'gain 1': gain_1, 'loss 1': loss_1, 'overage 1': account_1, 
                               'model reply 2': reply_2, 'selection 2': selection_2, 'gain': gain, 'loss': loss, 'overage': account}

            except Exception as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'selection 1': None, 'gain 1': None, 'loss 1': None, 'overage 1': None, 'selection 2': None, 'gain': None, 'loss': None, 'overage': None}  
            
            if pbar:
                pbar.update(1)
            time.sleep(0)
        
        return outcomes,conversation
        
    def test_dcigt(self):
        all_sessions_outcomes = {}
        conversations = {}
        total = self.sessions * self.trials
        with tqdm (total=total, desc='Testing Double-Choice Iowa Gambling Test: ') as pbar:

            for session in range(self.sessions):
                outcomes, conversation = self.dcigt(pbar)
                all_sessions_outcomes[session] = outcomes
                conversations[session] = conversation
            return all_sessions_outcomes, conversations
        
    def evaluate_dcigt(self):
        results_dcigt, conversations_dcigt = self.test_dcigt()
        
        try:
            dcigt_scorer = DCIGTSCORER(self.sessions, self.trials, results_dcigt)
            score_dcigt = dcigt_scorer.scoring_dcgit()
            print(f"Score of Double-Choice Iowa Gambling Task of {self.model}: ", score_dcigt)
        
        except Exception as e:
            print(f"Error occurred: {str(e)}.")
            score_dcigt = None
        
        return results_dcigt, conversations_dcigt, score_dcigt
