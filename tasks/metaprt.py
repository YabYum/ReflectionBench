import numpy as np
import time

class MetaPRT:
    def __init__(self, client):
        self.client = client

    def decision(self, model_name, instruction, conversation_history=None):
        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are playing a two-armed bandit game. Each time you need to choose between the right arm (AAA) or the left arm (BBB). You will receive a feedback (0 or 1) based on your choice. Your goal is to maximize the total reward. Respond only 'AAA' or 'BBB' without outputing anything else. keep performing the task until the end of the test."}]

        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 1
        )
        
        reply = completion.choices[0].message.content.replace('\n', '')

        conversation_history.append({"role": "assistant", "content": reply})

        if 'AAA' in reply.upper():
            return 'AAA', conversation_history
        elif 'BBB' in reply.upper():
            return 'BBB', conversation_history
        else:
            raise ValueError(f"Invalid response from model: {reply}")

    def instruction(self, trail_no, prior_reward):
        if trail_no == 1:
            instruction = "This is the first trial, choose one arm please."
        else:
            instruction = f"Your previous choice resulted in a reward of {prior_reward}. This is trial NO. {trail_no}. Please choose the option you think is best."

        return instruction

    def get_reward(self, side, p):
        q = 1 - p
        if side == 'AAA':
            reward = np.random.binomial(n=1, p=p, size=1)[0] # AAA for right, therefore p is the probability of getting reward of right arm
        elif side == 'BBB':
            reward = np.random.binomial(n=1, p= q, size=1)[0] # BBB for left
        
        return reward

    def metaprt(self, model_name, trials, p, interval_1, interval_2):
        conversation = None
        outcomes = {}
        reward = 0
        half_trials = trials // 2
        
        for i in range(trials):
            try:

                if i < half_trials:
                    interval = interval_1
                    mark = i
                else:
                    interval = interval_2
                    mark = i - half_trials
            
                if i>0 and mark%interval == 0: 
                    p=1-p

                trial_no = i + 1
                inst = self.instruction(trial_no, reward)
                decision, conversation = self.decision(model_name, inst, conversation)
                reward = self.get_reward(decision, p)
                outcomes[trial_no] = {'decision': decision, 'reward': reward}

            except (RuntimeError, ValueError) as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'decision': 'ERROR', 'reward': None}
            time.sleep(0)

        return outcomes, conversation


    def test_metaprt(self, model_name, trials, p, sessions, interval_1, interval_2):
        all_sessions_outcomes = {}
        conversations = []
        for session in range(sessions):
            outcomes, conversation = self.metaprt(model_name, trials,p, interval_1, interval_2)
            all_sessions_outcomes[session + 1] = outcomes
            conversations.append(conversation)
        return all_sessions_outcomes, conversations
