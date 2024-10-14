import numpy as np
import time


class IGT:
    def __init__(self,client):
        self.client = client
        self.p_losses = {'AAA':0.5, 'BBB':0.1, 'CCC':0.5, 'DDD':0.1}
        self.num_losses = {'AAA':260, 'BBB':1250, 'CCC':50, 'DDD':200}
        self.num_gains = {'AAA':100, 'BBB':100, 'CCC':50, 'DDD':50}

    def choose(self,model_name,instruction,conversation_history=None):

        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are participating in an interesting game. There are four decks of cards in front of you, labeled 'AAA', 'BBB', 'CCC', and 'DDD'. You start with $2000 in your account. Your goal is to maximize your account assets by making a series of choices from these decks. For each trial: 1. Choose one deck (AAA, BBB, CCC, or DDD). 2. You will receive feedback on the outcome of your choice, including any gains and losses. 3. Your account balance will be updated accordingly. Respond only with your deck choice ('AAA', 'BBB', 'CCC', or 'DDD') for each trial without any explanations or punctuations. Continue making choices until instructed to stop. "}]

        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 0
        )
        
        reply = completion.choices[0].message.content.replace('\n', '')

        conversation_history.append({"role": "assistant", "content": reply})

        if 'AAA' in reply.upper():
            return 'AAA', conversation_history
        elif 'BBB' in reply.upper():
            return 'BBB', conversation_history
        elif 'CCC' in reply.upper():
            return 'CCC', conversation_history
        elif 'DDD' in reply.upper():
            return 'DDD', conversation_history
    
    def instruction(self,trial_no,overage,gain=None,loss=None):

        if gain is None: 
            instruction = f"The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        else:
            instruction = f"The outcome of your last choice is gain: ${gain}, loss: ${loss}. The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        return instruction
    
    def outcome(self,choice):
        p_loss = self.p_losses[choice]
        num_loss = self.num_losses[choice]
        loss =  np.random.binomial(n=1, p=p_loss, size=1)[0] * num_loss
        gain = self.num_gains[choice]
        return loss, gain
    
    def igt(self,model_name,trials):
        conversation = None
        gain = None
        loss = None
        account = 2000
        outcomes = {}

        for i in range(trials):
            try:
                trial_no = i+1
                instruction = self.instruction(trial_no, account, gain, loss)
                selection, conversation = self.choose(model_name, instruction, conversation)
                loss, gain = self.outcome(selection)
                account += (gain - loss)

                outcomes[i] = {'selection': selection, 'gain': gain, 'loss': loss, 'overage': account}
            except Exception as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'selection': None, 'gain': None, 'loss': None, 'overage': None}
            time.sleep(0)
             
        return outcomes,conversation
        
    def test_igt(self,model_name,trials,sessions):
        all_sessions_outcomes = {}
        conversations = []
        for session in range(sessions):
            outcomes, conversation = self.igt(model_name, trials)
            all_sessions_outcomes[session] = outcomes
            conversations.append(conversation)
        return all_sessions_outcomes, conversations    

