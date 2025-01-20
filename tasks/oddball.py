import json
from utils.subject import Subject
from scorer.oddball_scorer import ODDBALLSCORER
from tqdm import tqdm

class Oddball:
    def __init__(self, key_subject, url_subject, model_name, config, COT=None):
        self.subject = Subject(key=key_subject, url=url_subject, task="oddball", model=model_name)     
        with open("dataset/prompts_oddball.json", 'r') as file:
            self.prompts = json.load(file)
        self.length = len(self.prompts)
        self.COT = COT
        self.sessions = config['oddball']['sessions']
        self.config = config
        self.model = model_name
        if COT == None:
            self.cot_instruction = " "
        elif COT == True:
            self.cot_instruction = " let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = " respond only with your comments directly without outputing any other infomation or analysis."
        
    
    def test_oddball(self, sessions):

        results = {}
        total_trials = sessions * self.length
        pbar = None

        with tqdm (total = total_trials, desc='Testing: Oddball Paradigm') as pbar:
            
            for session in range(sessions):
                comments = {}
                for i in range(len(self.prompts)):
                    try:
                        prompt = self.prompts[f"{i+1}"] + self.cot_instruction
                        comment, conversation = self.subject.conversation(prompt)
                        comments[i] = comment

                    except Exception as e:
                        comments[i] = None
                        print(f"ERROR: {e}, skipping this trial {i}.")  

                    pbar.update(1)         
                results[f"session_{session}"] = comments

        return results

    def evaluate_oddball(self):

        results = self.test_oddball(self.sessions)
        scoreroddball = ODDBALLSCORER(self.config)
        try:
            scores = scoreroddball.scoreringoddball(results, self.sessions, self.length)
            print(f"Score of Oddball paradigm of {self.model}: ", scores)

        except Exception as e:
            scores = None
            print(f"ERROR: {e} while scoring.")

        return results, scores
