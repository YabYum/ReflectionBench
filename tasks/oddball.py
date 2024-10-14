import json

class Oddball:
    def __init__(self, client):
        self.client = client        
        with open("dataset/prompts_oddball.json", 'r') as file:
            self.prompts = json.load(file)
    
    def reply(self, model_name, prompt):

        role_sys = 'user' if model_name in ["o1-preview", "o1-mini"] else "system"
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": role_sys, "content": "You are playing a game and will be presented with a sequence of sentences about specific topic. Just make short comments on the material."},
                      {"role": "user", "content": f"{prompt}"}],
            temperature = 1
        )

        reply = completion.choices[0].message.content

        return reply
    
    def test_oddball(self, model_name, sessions):

        results = {}

        for session in range(sessions):

            comments = {}

            for i in range(len(self.prompts)):
                prompt = self.prompts[f"{i+1}"]
                comment = self.reply(model_name, prompt)
                comments[i+1] = comment
                print(f"number: {i}")
            
            results[f"session_{session+1}"] = comments

        with open(f"{model_name}-comments.json", 'w') as f:
            json.dump(results, f, indent=4)

        return results
    