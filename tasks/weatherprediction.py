from utils.extractor import EXTRACTOR
from utils.subject import Subject
import numpy as np
import time
from tqdm import tqdm
from scorer.wpt_scorer import WPTSCORER


class WeatherPred:


    def __init__(self, key_subject, url_subject, model_name, config, COT=None):
         
        self.extractor = EXTRACTOR(task="weather prediction task_extractor")
        self.subject = Subject(key=key_subject, url=url_subject, task="weather prediction task", model=model_name)
        with open("dataset/weather.txt") as file:
            self.devices = [line.strip() for line in file]
        self.transition_1 = config['wpt']['transition_1'] 
        self.transition_2 = config['wpt']['transition_2']
        self.states = ['sunny','rainy']
        self.states_index = {'sunny':0, 'rainy':1}
        self.transition_index = {'[1, 0]': self.transition_1, '[0, 1]': self.transition_2}
        self.sessions = config['wpt']['sessions']
        self.estimate_interval = config['wpt']['estimate_interval']
        self.model = model_name
        if COT == None:
            self.cot_instruction = ""
        elif COT == True:
            self.cot_instruction = "let's think step by step."   
        elif COT == "Direct":
            self.cot_instruction = "respond only with your choice directly without outputing any other infomation or analysis"
    

    def transition(self, today_weather, device_state):

        """calculate the tommorow's real weather based on the transition matrix"""

        transition_matrix = self.transition_index[device_state]
        probability = transition_matrix[self.states_index[today_weather]]
        tommorow_weather = np.random.choice(self.states, p=probability)

        return tommorow_weather
    
    def instruction(self,index,today_weather,outcomes):

        trial_no = index + 1
        device_state = self.devices[index]

        if index > 0: 
            prediction = outcomes[index - 1]['model prediction']

        if trial_no == 1: 
            instruction = f"this is the first day of your work. today's weather is sunny, and the state of devices is {device_state}. Now try to make your prediction on tomorrow's weather (just predict sunny or rainy)! "
        else:
            instruction = f"This is the next day now, your prediction on today's weather was {prediction}, and today's actual weather is {today_weather}. Today, the state of devices is {device_state}, now make your prediction according to today's weather and the state of devices (just predict sunny or rainy). "
        
        return device_state, instruction + self.cot_instruction

    def wpt(self, pbar=None):

        weathers = ['sunny']
        conversation = None
        outcomes = {}

        for i in range(len(self.devices)):
        
            today_weather = weathers[i]
            device_state, instruction = self.instruction(i, today_weather, outcomes)
            tomorrow_weather = self.transition(today_weather, device_state)
            weathers.append(tomorrow_weather)
            try:
                reply, conversation = self.subject.conversation(instruction, conversation)
                prediction = self.extractor.extraction(reply)
                outcomes[i] = {'today weather': today_weather, 'device state': device_state, 'model reply': reply, 'model prediction': prediction, 'tomorrow weather': tomorrow_weather}

            except Exception as e:
                print(f"Error in trial {i}: {e}")
                outcomes[i] = {'today weather': today_weather, 'device state': device_state, 'model reply': 'Not available', 'model prediction': 'Not available', 'tomorrow weather': tomorrow_weather}

            if pbar:
                pbar.update(1)
            time.sleep(0)

        return outcomes, conversation
    
    def test_wpt(self):
        all_session_outcomes = {}
        conversations = {}
        total_trials = len(self.devices) * self.sessions
        
        with tqdm (total=total_trials, desc='Testing Weather Prediction Task: ') as pbar:

            for session in range(self.sessions):
                outcomes, conversation = self.wpt(pbar)
                all_session_outcomes[session] = outcomes
                conversations[session] = conversation

        return all_session_outcomes, conversations
    
    def evaluate_wpt(self):

        results_wpt, conversations_wpt = self.test_wpt()
        
        try:
            scorer_wpt = WPTSCORER(self.sessions, results_wpt, self.estimate_interval)
            score_wpt = scorer_wpt.scoring_wpt()
            print(f"Score of Weahter Prediction Task of {self.model}: ", score_wpt)

            return results_wpt, conversations_wpt, score_wpt
        except Exception as e:
            print(f"Error occurred: {str(e)}.")
            return results_wpt, conversations_wpt, None
