import numpy as np
import time


class DoubleChoiceIGT:
    def __init__(self,client):
        self.client = client
        self.p_loss = {'AAA':0.5, 'BBB':0.1, 'CCC':0.5, 'DDD':0.1}
        self.num_loss = {'AAA':260, 'BBB':1250, 'CCC':50, 'DDD':200}
        self.num_gain = {'AAA':100, 'BBB':100, 'CCC':50, 'DDD':50}

    def choose(self,model_name,instruction,conversation_history=None):
        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are participating in an interesting game. There are four decks of cards in front of you, labeled 'AAA', 'BBB', 'CCC', and 'DDD'. You start with $2000 in your account. Your goal is to maximize your account assets by making a series of choices from these decks. For each trial: 1. Choose one deck ('AAA', 'BBB', 'CCC', or 'DDD'). 2. You will receive feedback on the outcome of your choice, including any gains and losses. 3. After reveiving the feedback, you have one opportunity to reconsider your decision, you can either stick with your original choice or make a new choice. 4. Your final choice would determin your actual gain or loss for the trial. Respond only with your deck choice (AAA, BBB, CCC, or DDD) for each choice without any explanations and punctuations. keep making choices."}]


        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 1
        )
        
        reply = completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": reply})

        return reply, conversation_history
        
    
    def instruction_1(self,trial_no,overage,gain=None,loss=None):
        if gain is None: 
            instruction = f"The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        else:
            instruction = f"The outcome of your last choice is gain: ${gain}, loss: ${loss}. The current money in your account is ${overage}. This is the NO. {trial_no} trial, make your choice!"
        return instruction
    
    def instruction_2(self,overage,gain=None,loss=None):

        instruction = f"If your choose this deck, the outcome will be gain: ${gain}, loss: ${loss}. the money in your account will ${overage}. Which deck would you choose? Make your final choice!"

        return instruction
    
    def outcome(self,choice):
        p_loss = self.p_loss[choice]
        num_loss = self.num_loss[choice]
        loss =  np.random.binomial(n=1, p=p_loss, size=1)[0] * num_loss
        gain = self.num_gain[choice]
        return loss, gain
    
    def dcigt(self,model_name,trials):
        conversation = None
        gain = None
        loss = None
        account = 2000
        outcomes = {}

        for i in range(trials):
            try:
                trial_no = i+1
                instruction_1 = self.instruction_1(trial_no, account, gain, loss)
                selection_1, conversation = self.choose(model_name, instruction_1, conversation)
                loss_1, gain_1 = self.outcome(selection_1)
                account_1 = account + gain_1 - loss_1
                instruction_2 = self.instruction_2(account_1,gain_1,loss_1)
                selection_2, conversation = self.choose(model_name, instruction_2, conversation)
                loss, gain = self.outcome(selection_2)
                account += (gain - loss)

                outcomes[i] = {'selection 1': selection_1, 'gain 1': gain_1, 'loss 1': loss_1, 'overage 1': account_1, 'selection 2': selection_2, 'gain': gain, 'loss': loss, 'overage': account}
                print(outcomes[i])
            except Exception as e:
                print(f"Error in trial {trial_no}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'selection 1': None, 'gain 1': None, 'loss 1': None, 'overage 1': None, 'selection 2': None, 'gain': None, 'loss': None, 'overage': None}  
            
            time.sleep(0)
        
        return outcomes,conversation
        
    def test_dcigt(self,model_name,trials,sessions):
        all_sessions_outcomes = {}
        conversations = []
        for session in range(sessions):
            outcomes, conversation = self.dcigt(model_name, trials)
            all_sessions_outcomes[session+1] = outcomes
            conversations.append(conversation)
        return all_sessions_outcomes, conversations
    

