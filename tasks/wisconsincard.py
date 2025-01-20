import json
import time
from utils.extractor import EXTRACTOR
from utils.subject import Subject
from tqdm import tqdm
from scorer.wcst_scorer import WCSTSCORER


class WCST:


    def __init__(self, key_subject, url_subject, model_name, config, COT=None):
        with open("dataset/cards.json") as file:
            self.cards = json.load(file)
        self.extractor = EXTRACTOR(task="wisconsin card sorting test_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="wisconsin card sorting test", model=model_name)
        self.rules = ['shape', 'color', 'number']
        self.sessions = config['wcst']['sessions']
        self.rule_change_interval = config['wcst']['rule_change_interval']
        self.model = model_name
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis"
    
    
    def instruction(self, trial_no, feedback=None):

        testingcard = self.cards[f"{trial_no}"]
        instruction = f"This is the first trial, the testing card is {testingcard}. " if trial_no == 1 else f"Your last choice was {feedback}. This is trial number {trial_no}, your testing card is {testingcard}. "
        
        return testingcard, instruction + self.cot_instruction
   
    def feedback(self, rule, test, option):
        if rule == 'shape':
            is_right = test.split()[0] == option.split()[0]
        elif rule == 'color':
            is_right = test.split()[1] == option.split()[1]
        elif rule =='number':
            is_right = test.split()[2] == option.split()[2]
        return is_right
    
    def wcst(self, pbar=None):
        
        conversation = None
        feedback = None
        outcomes = {}
        rule_index = 0


        for i in range(len(self.cards)):
            try:
                if i != 0 and i % self.rule_change_interval == 0:rule_index = (rule_index+1)%3
                current_rule = self.rules[rule_index]

                trial_no = i+1
                testingcard, instruction = self.instruction(trial_no,feedback)

                reply, conversation = self.subject.conversation(instruction,conversation)
                option = self.extractor.extraction(reply)

                feedback = self.feedback(current_rule, testingcard, option)

                outcomes[i] = {'rule': current_rule, 'testing card': testingcard, 'reply': reply, 'option':option, 'feedback': feedback}

            except Exception as e:
                print(f"Error occurred in trial {i + 1}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'rule': current_rule, 'testing card': testingcard, 'option':option, 'feedback': str(e), 'error': True}
                feedback = None
            
            if pbar:
                pbar.update(1)

            time.sleep(0)
            
        return outcomes, conversation
    
    def test_wcst(self):
        all_session_outcomes = {}
        conversations = {}
        total = len(self.cards) * self.sessions

        with tqdm (total=total, desc='Testing Wisconsin Card Sorting Test: ') as pbar:
            for session in range(self.sessions):
                outcomes, conversation = self.wcst(pbar)
                all_session_outcomes[session]=outcomes
                conversations[session]=conversation
            
        return all_session_outcomes, conversations
    
    def evaluate_wcst(self):
        results_wcst, conversations_wcst = self.test_wcst()

        try:
            scorer = WCSTSCORER(self.sessions)
            score_wcst = scorer.scoringwcst(results_wcst)
            print(f"Score of Wisconsin Card Sorting Test of {self.model}: ", score_wcst)            
        
        except Exception as e:
            print(f"Error occurred: {str(e)}.")
            score_wcst = None
        
        return results_wcst, conversations_wcst, score_wcst



