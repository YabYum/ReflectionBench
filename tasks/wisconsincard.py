import json
import time


class WCST:
    def __init__(self, client):
        self.client = client
        self.rules = ['shape', 'color', 'number']
        with open("dataset/cards.json") as file:
            self.cards = json.load(file)

    def choose(self, model_name, instruction, conversation_history=None):

        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are performing an interesting Task. In this task, you have four cards on your desk, that is, 'triangle red 1', 'cross green 2', 'circle yellow 1', and 'star blue 4'. The three word/figure represent the the type of shape, i.e. triangle, cross, circle, or star, the color of the shape, i.e. red, green, yellow, or blue, and the number of the shape, i.e., 1, 2, 3, or 4, respectively. At each trial, you will be presented with a testing card. You shoud point out which card on your desk matches the testing card. I will not tell you the matching rule but only provide feedback if your choice was right or wrong. Your primary goal is strive to maximize your accuracy rate. Respond only with your option ('triangle red 1', 'cross green 2', 'circle yellow 1', or 'star blue 4') without outputing any other information. keep performing the task until the end of the test. "}]

        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 0
        )
        
        reply = completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": reply})

        return reply, conversation_history
    
    def instruction(self, trial_no, feedback=None):
        testingcard = self.cards[f"{trial_no}"]
        if trial_no == 1:
            instruction = f"This is the first trial, the testing card is {testingcard}"
        else:
            feedback_str = "correct" if feedback == True else "incorrect"
            instruction = f"Your last choice was {feedback_str}. This is trial number {trial_no}, your testing card is {testingcard}"
        return testingcard, instruction
   
    def feedback(self, rule, test, option):
        if rule == 'shape':
            is_right = test.split()[0] == option.split()[0]
        elif rule == 'color':
            is_right = test.split()[1] == option.split()[1]
        elif rule =='number':
            is_right = test.split()[2] == option.split()[2]
        return is_right
    
    def wcst(self,model_name,rule_change_interval):
        
        conversation = None
        feedback = None
        outcomes = {}
        rule_index = 0


        for i in range(len(self.cards)):
            try:
                if i != 0 and i % rule_change_interval == 0:rule_index = (rule_index+1)%3
                current_rule = self.rules[rule_index]

                trial_no = i+1
                testingcard, instruction = self.instruction(trial_no,feedback)

                option, conversation = self.choose(model_name,instruction,conversation)
                feedback = self.feedback(current_rule, testingcard, option)

                outcomes[i] = {'rule': current_rule, 'testing card': testingcard, 'option':option, 'feedback': feedback}

            except Exception as e:
                print(f"Error occurred in trial {i + 1}: {str(e)}. Skipping this trial.")
                outcomes[i] = {'rule': current_rule, 'testing card': testingcard, 'option':option, 'feedback': str(e), 'error': True}
                feedback = None
            
            time.sleep(0)
            

        return outcomes, conversation
    
    def test_wcst(self,model_name,rule_change_interval,sessions):
        all_session_outcomes = {}
        conversations = []

        for session in range(sessions):
            outcomes, conversation = self.wcst(model_name,rule_change_interval)
            all_session_outcomes[session]=outcomes
            conversations.append(conversation)
            
        return all_session_outcomes, conversations
