import numpy as np
import re
from openai import OpenAI
from tqdm import tqdm

class ODDBALLSCORER:

    def __init__(self, CONFIG):
        self.client = OpenAI(api_key=CONFIG['api_setting']['api_key_embedding'], base_url=CONFIG['api_setting']['base_url_embedding'])

    def get_embedding(self, input):
        return self.client.embeddings.create(input=input, model="text-embedding-3-large").data[0].embedding

    def split(self,answer):
        return re.split(r'(?<=[.!?])\s+', answer)

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def calculate_surprisal(self,sentences):
        context = "Wait, what? It is a totally unexpected shift and doesn't make any sense! It is not related to the context we are talking about! I don't get it. It is completely confusing."
        context_embedding = self.get_embedding(context)
        simialrity = []
        for sentence in sentences:
            sentence_embedding = self.get_embedding(sentence)
            simialrity_embedding = self.cosine_similarity(context_embedding, sentence_embedding)
            simialrity.append(simialrity_embedding)    
        return simialrity

    def scoreringoddball(self, results, sessions, length):

        total = sessions * length    
        surprisal = []

        with tqdm (total=total, desc='Scoring Oddball: ') as pbar:
            for session in range(sessions):
                for trial in range(length):
                    try:
                        answer = results[f"session_{session}"][trial]
                        similarity = self.calculate_surprisal(self.split(answer))
                        surprisal.append(np.max(similarity))

                    except Exception as e:
                        print(f"Encounter error: {e}, skip scoring this answer sesseion_{session}, trial_{trial}.")
                        
                    pbar.update(1)
        
        return np.sum(surprisal)
    