import time
from utils.extractor import EXTRACTOR
from utils.subject import Subject
from tqdm import tqdm
from scorer.nback_scorer import NBACK_SCORER

class Nback:

    def __init__(self, back_n, key_subject, url_subject, model_name, config, COT=None):
         
        self.back_n = back_n
        with open("dataset/n-back-seq.txt", 'r') as file:
            self.sequence = file.read().strip()
        self.extractor = EXTRACTOR(task="n-back_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="n-back", model=model_name)
        self.config = config
        self.model = model_name
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis."

    def nback(self, pbar=None):
        sequence = self.sequence
        conversation_history = None     
        results = {}
        
        for i, char in enumerate(sequence):
            
            # present the current, instruct the model to judge if it is the same as the one two-steps before.
            instruction = f"Current character is: {char}, make your decision. " if i == 0 else f"Your answer is {is_correct}. Current character is: {char}, make your decision. " 
            instruction = instruction + self.cot_instruction
            
            # get the model's response
            decision, conversation_history = self.subject.conversation(instruction, conversation_history)

            # extract model's response and transform into simple marker
            reply = self.extractor.extraction(decision)

            # expected answer
            if i < self.back_n: correct_answer = "CCC"
            elif char == sequence[i-self.back_n]: correct_answer = "AAA"
            else: correct_answer = "BBB"

            # check the model's response
            is_correct = correct_answer in reply

            results[i] = {"stimuli": char, "model_answer": decision, "extraction": reply, "correct_answer": correct_answer, "is_correct": is_correct}
            if pbar:
                pbar.update(1)
            time.sleep(0)
        
        return results, conversation_history
    
    def test_nback(self):

        results = {}
        conversations = {}
        sessions = self.config['nback']['sessions']
        total_trials = sessions * len(self.sequence)

        with tqdm (total=total_trials, desc='Testing: N-BACK') as pbar:
            for session in range(sessions):
                result, conversation = self.nback(pbar)
                results[session] = result
                conversations[session] = conversation
        
        return results, conversations, len(self.sequence)
    
    def evaluate_nback(self):
       
        all_outcomes_nback, conversations_nback, len_squence_nback = self.test_nback()
        scorer_nback = NBACK_SCORER(all_outcomes_nback, self.config['nback']['sessions'],len_squence_nback)
        score_nback = scorer_nback.scoringnback()
        print(f"Score of N-Back Task of {self.model}: ", score_nback)

        return all_outcomes_nback, conversations_nback, score_nback
    