import numpy as np
import time


class WeatherPred:
    def __init__(self,client):
        self.client = client
        with open("dataset/weather.txt") as file:
            self.devices = [line.strip() for line in file]
        self.transition_1 = [
            [0.9 , 0.1],
            [0.1 , 0.9]
            ]
        self.transition_2 = [
            [0.1, 0.9], 
            [0.9, 0.1]
            ]
        self.states = ['sunny','rainy']
        self.states_index = {'sunny':0, 'rainy':1}
        self.transition_index = {'[1, 0]': self.transition_1, '[0, 1]': self.transition_2}
    
    def prediction(self,model_name,instruction,conversation_history=None):
        
        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are an expert forecaster working in a weather station. There are two devices collecting data from nature. Your task is to predict tomorrow's weather based on today's weather and the current states of four sensor devices in the weather station. Here's how the task works: 1. There are two devices, each represented by either 0 (inactive) or 1 (active). 2. The device states will be given to you in the format [d1,d2], where each d is either 0 or 1; 3. Based on these device states and today's weather, you need to predict whether tomorrow's weather will be sunny or rainy. 4. After your prediction, I will inform you of the actual weather outcome. 5. We will repeat this process multiple times, and you should try to improve your predictions based on the feedback. At each time, just make your prediction ('sunny' or 'rainy'), without outputing any other information. keep performing the task until the end of the test."}]

        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 0
        )
        
        reply = completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": reply})

        return reply, conversation_history

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
            instruction = f"this is the first day of your work. today's weather is sunny, and the state of devices is {device_state}. Now try to make your prediction on tomorrow's weather (just predict sunny or rainy)!"
        else:
            instruction = f"This is the next day now, your prediction on today's weather was {prediction}, and today's actual weather is {today_weather}. Today, the state of devices is {device_state}, now make your prediction according to today's weather and the state of devices (just predict sunny or rainy)."
        return device_state, instruction

    def wpt(self,model_name):

        weathers = ['sunny']
        conversation = None
        outcomes = {}

        for i in range(len(self.devices)):
            today_weather = weathers[i]
            device_state, instruction = self.instruction(i, today_weather, outcomes)
            prediction, conversation = self.prediction(model_name, instruction, conversation)
            tomorrow_weather = self.transition(today_weather, device_state)
            weathers.append(tomorrow_weather)
            outcomes[i] = {'today weather': today_weather, 'device state': device_state, 'model prediction': prediction, 'tomorrow weather': tomorrow_weather}
            print(f"trial{i}: ", outcomes[i])
            time.sleep(0)

        return outcomes, conversation
    
    def test_wpt(self,model_name,sessions):
        all_session_outcomes = {}
        conversations = []

        for session in range(sessions):
            print(f"session{session}")
            outcomes, conversation = self.wpt(model_name)
            all_session_outcomes[session+1] = outcomes
            conversations.append(conversation)

        return all_session_outcomes, conversations
