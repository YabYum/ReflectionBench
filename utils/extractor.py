from openai import OpenAI
import json
from config import CONFIG

class EXTRACTOR:
    def __init__(self, task):
        
        self.client = OpenAI(api_key= CONFIG['api_setting']['api_key_dpsk'], base_url=CONFIG['api_setting']['base_url_dpsk'])

        with open("dataset/systemprompts.json", 'r') as file:
            self.sysprompt = json.load(file)[task]

    def extraction(self, input):
        response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.sysprompt},
                    {"role": "user", "content": input}],
                    temperature=0
                )

        extraction = response.choices[0].message.content
        
        return extraction
