import time
class Nback:

    def __init__(self, client, back_n):
        self.client = client
        self.back_n = back_n        
        with open("dataset/n-back-seq.txt", 'r') as file:
            self.sequence = file.read().strip()

    def judge(self, model_name, instruction, conversation_history=None):
        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"

        if conversation_history is None:
            conversation_history = [{"role": role_sys, "content": "You are playing a game. I will give you a series of characters in sequence, showing only one at a time. Your task is to determine whether the current character is the same as the character 2 steps before. Rules: If the current character is the same as the character 2 steps before, answer 'AAA'. If the current character is different from the character 2 steps before, answer 'BBB'. For the first 2 steps, since there aren't enough preceding characters for comparison, answer 'CCC'. Provide only your judgment (AAA, BBB, or CCC), without explanation or additional information."}]
        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature = 0
        )
        
        reply = completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": reply})

        return reply, conversation_history

    def nback(self, model_name):
        sequence = self.sequence
        conversation_history = None
        results = {}
        
        for i, char in enumerate(sequence):
            instruction = f"Current character: {char}"
            reply, conversation_history = self.judge(model_name, instruction, conversation_history)

            if i < self.back_n: correct_answer = "CCC"
            elif char == sequence[i-self.back_n]: correct_answer = "AAA"
            else: correct_answer = "BBB"

            is_correct = correct_answer in reply

            results[i] = {"stimuli": char, "model_answer": reply, "correct_answer": correct_answer, "is_correct": is_correct}
            time.sleep(0)
        
        return results, conversation_history
    
    def test_nback(self, model_name, sessions):

        results = {}
        conversations = {}

        for session in range(sessions):
            result, conversation = self.nback(model_name)
            results[session] = result
            conversations[session] = conversation
        
        return results, conversations
