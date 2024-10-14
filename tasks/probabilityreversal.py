import numpy as np
import time

class PRT:
    """Probabilistic Reversal Task"""

    def __init__(self, client):
        self.client = client

    def decision(self, model_name, instruction, conversation_history=None):
        
        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are playing a two-armed bandit game. Each time you need to choose between the right arm (AAA) or the left arm (BBB). You will receive a feedback (0 or 1) based on your choice. Your goal is to maximize the total reward. Respond only AAA or BBB without outputing anything else. keep performing the task until the end of the test."}]

        conversation_history.append({"role": "user", "content": instruction})
        
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                temperature = 0
            )
        
            reply = completion.choices[0].message.content

            conversation_history.append({"role": "assistant", "content": reply})

            if 'AAA' in reply.upper():
                return 'AAA', conversation_history
            elif 'BBB' in reply.upper():
                return 'BBB', conversation_history
            else:
                raise ValueError(f"Invalid response from model: {reply}")
        
        except Exception as e:
            raise RuntimeError(f"Error during decision generation: {str(e)}")

    def instruction(self, trail_no, prior_reward):
        if trail_no == 1:
            instruction = "This is the first trial, choose one arm please."
        else:
            instruction = f"Your previous choice resulted in a reward of {prior_reward}. This is trial number {trail_no}. Please choose the option you think is best."

        return instruction

    def reward(self, side, p):
        return np.random.binomial(n=1, p=p if side == 'AAA' else 1-p, size=1)[0]

    def prt(self, model_name, trials, p):
        conversation = None
        outcomes = {}
        reward = 0
        rp = 1 - p
        
        for i in range(trials):
            trial_no = i + 1
            instruction = self.instruction(trial_no, reward)
            try:
                decision, conversation = self.decision(model_name, instruction, conversation)
                reward = self.reward(decision, p if i < trials/2 else rp)
                outcomes[i] = {'decision': decision, 'reward': reward}
            except (RuntimeError, ValueError) as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'decision': 'ERROR', 'reward': None}
    
            time.sleep(0)
        

        return outcomes, conversation

    def test_prt(self, model_name, trials, p, sessions):
        all_sessions_outcomes = {}
        conversations = []

        for session in range(sessions):
            outcomes, conversation = self.prt(model_name, trials, p)            
            all_sessions_outcomes[session] = outcomes
            conversations.append(conversation)
        
        return all_sessions_outcomes, conversations
    
    