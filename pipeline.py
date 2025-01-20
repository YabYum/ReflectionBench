from tasks.nback import Nback
from tasks.oddball import Oddball
from tasks.probabilityreversal import PRT
from tasks.wisconsincard import WCST
from tasks.weatherprediction import WeatherPred
from tasks.doublechoiceigt import DoubleChoiceIGT
from tasks.metaprt import MetaPRT
from config import CONFIG
import pickle
import os

class REFLECTIONBENCH:
     
    def __init__(self, model, COT=None):
        self.model = model
        self.config = CONFIG
        self.COT = COT

    def update_pickle(self, task_name, outcomes, conversations, scores=None):
        
        pickle_file = f"{self.model}_results_{self.COT}.pkl"

        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                model_results = pickle.load(f)
        else:
            model_results = {}

        if scores==None:
            model_results[task_name] = {
                'outcomes': outcomes,
                'conversations': conversations
        }
        
        else: 
            model_results[task_name] = {
                'outcomes': outcomes,
                'conversations': conversations,
                'score': scores
        }

        with open(pickle_file, 'wb') as f:
            pickle.dump(model_results, f)

    def evaluate(self):
        # Oddball Paradigm
        oddball = Oddball(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=self.config, COT=self.COT)
        results_odb, score_oddball = oddball.evaluate_oddball()
        self.update_pickle('oddball', results_odb, "NO CONVERSATION", score_oddball)
        
        # N-Back
        nback = Nback(back_n=2, key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=self.config, COT=self.COT)        
        outcomes_nback, conversations_nback, score_nback = nback.evaluate_nback()
        self.update_pickle('2back', outcomes_nback, conversations_nback, score_nback)

        # Probabilistic Reversal Learning Task
        prlt = PRT(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=self.config, COT=self.COT)
        results_prlt, conversations_prlt, score_prlt = prlt.evaluate_prlt()
        self.update_pickle('PRLT', results_prlt, conversations_prlt, score_prlt)
        
        # Wisconsin Card Sorting Test
        wcst = WCST(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=CONFIG, COT=self.COT)
        results_wcst, conversations_wcst, score_wcst = wcst.evaluate_wcst()
        self.update_pickle('WCST', results_wcst, conversations_wcst, score_wcst)

        # Weather Prediction Task
        wpt = WeatherPred(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=CONFIG, COT=self.COT)
        results_wpt, conversations_wpt, score_wpt = wpt.evaluate_wpt()
        self.update_pickle('WPT', results_wpt, conversations_wpt, score_wpt)

        # Double-Choice Iowa Gambling Test
        dcigt = DoubleChoiceIGT(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=CONFIG, COT=self.COT)
        results_dcigt, conversations_dcigt, score_dcigt = dcigt.evaluate_dcigt()
        self.update_pickle('DCIGT',results_dcigt, conversations_dcigt, score_dcigt)

        # Meta-bandit Task
        metaprt = MetaPRT(key_subject=self.config['api_setting']['api_key'], url_subject=self.config['api_setting']['base_url'], model_name=self.model, config=CONFIG, COT=self.COT)
        results_metaprt, conversations_metaprt, score_metaprt = metaprt.evaluate_metaprt()
        self.update_pickle('MBT',results_metaprt, conversations_metaprt, score_metaprt)
