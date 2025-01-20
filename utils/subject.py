from openai import OpenAI
import json

class Subject:
    def __init__(self, key, url, task, model):
        
        self.client = OpenAI(api_key= key, base_url= url)
        self.model = model
        with open("dataset/systemprompts.json", 'r') as file:
            self.sysprompt = json.load(file)[task]

    def conversation(self, instruction, conversation_history=None):

        # o1-preview and o1-mini cannot set the 'system' role, so we replace it with 'user' for the two models
        role_sys = 'user' if self.model in ["o1-preview", "o1-mini"] else "system" 

        if conversation_history is None:

            conversation_history = [{"role": role_sys, "content": self.sysprompt}]

        conversation_history.append({"role": "user", "content": instruction})

        completion = self.client.chat.completions.create(
                model=self.model,
                messages= conversation_history,
                temperature=1
        )

        reply = completion.choices[0].message.content        
        conversation_history.append({"role": "assistant", "content": reply})

        return reply, conversation_history
