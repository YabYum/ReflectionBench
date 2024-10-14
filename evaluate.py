import pickle
from tasks import metaprt, nback, probabilityreversal, weatherprediction, wisconsincard, iowagambling, doublechoiceigt
from openai import OpenAI
from config import CONFIG


import pickle
import os

class Evaluate:
    def __init__(self):
        # setting API
        self.client = OpenAI(api_key= "", base_url="",)
        # evaluated model
        self.models = []
        self.config = CONFIG


    def update_pickle(self, model_name, task_name, outcomes, conversations):
        pickle_file = f"{model_name}_results.pkl"
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                model_results = pickle.load(f)
        else:
            model_results = {}

        model_results[task_name] = {
            'outcomes': outcomes,
            'conversations': conversations
        }

        with open(pickle_file, 'wb') as f:
            pickle.dump(model_results, f)

    def setting_tasks(self, model_name):

        # N-back, n = 2
        eval_nback = nback.Nback(self.client, back_n=self.config['nback_1']['back_n'])
        print(f"Evaluating model: {model_name}, task: n-back, n=2")
        all_outcomes_nback, conversations_nback = eval_nback.test_nback(model_name, self.config['nback_1']['sessions'])
        self.update_pickle(model_name, '2back', all_outcomes_nback, conversations_nback)
        
        # probability reversal task, p = 0.9
        eval_prt = probabilityreversal.PRT(self.client)
        print(f"Evaluating model: {model_name}, task: probability reversal task, p =0.9")
        all_outcomes_prt, conversations_prt = eval_prt.test_prt(model_name, self.config['prt_1']['trials'], self.config['prt_1']['p'], self.config['prt_1']['sessions'])
        self.update_pickle(model_name, 'prt', all_outcomes_prt, conversations_prt)

        # Wisconsin card sorting test
        eval_wcst = wisconsincard.WCST(self.client)
        print(f"Evaluating model: {model_name}, task: Wisconsin card sorting test")
        all_outcomes_wcst, conversations_wcst = eval_wcst.test_wcst(model_name, self.config['wcst']['rule_change_interval'], self.config['wcst']['sessions'])
        self.update_pickle(model_name, 'wcst', all_outcomes_wcst, conversations_wcst)

        # weather prediction task
        eval_wpt = weatherprediction.WeatherPred(self.client)
        print(f"Evaluating model: {model_name}, task: weather prediction task")
        all_outcomes_wpt, conversations_wpt = eval_wpt.test_wpt(model_name, self.config['wpt']['sessions'])
        self.update_pickle(model_name, 'wpt', all_outcomes_wpt, conversations_wpt)

        # Iowa gambling task
        eval_igt = iowagambling.IGT(self.client)
        print(f"Evaluating model: {model_name}, task: Iowa gambling task")
        all_outcomes_igt, conversations_igt = eval_igt.test_igt(model_name, self.config['igt']['trials'], self.config['igt']['sessions'])
        self.update_pickle(model_name, 'igt', all_outcomes_igt, conversations_igt)

        # double choice IGT
        eval_dcigt = doublechoiceigt.DoubleChoiceIGT(self.client)
        print(f"Evaluating model: {model_name}, task: double choice igt")
        all_outcomes_dcigt, conversations_dcigt = eval_dcigt.test_dcigt(model_name, self.config['dcigt']['trials'], self.config['dcigt']['sessions'])
        self.update_pickle(model_name, 'dcigt', all_outcomes_dcigt, conversations_dcigt)

        # meta-prt, interval = 3, p = 1
        eval_metaprt = metaprt.MetaPRT(self.client)
        print(f"Evaluating model: {model_name}, task: meta-prt, interval = 4")
        all_outcomes_metaprt, conversations_metaprt = eval_metaprt.test_metaprt(model_name, self.config['metaprt_1']['trials'], self.config['metaprt_1']['p'], self.config['metaprt_1']['sessions'], self.config['metaprt_1']['interval_1'],self.config['metaprt_1']['interval_2'])
        self.update_pickle(model_name, 'metaprt_interval4', all_outcomes_metaprt, conversations_metaprt)


    def evlaute_reflection_tests(self):
        for model in self.models:
            print(f"Evaluating model: {model}")
            self.setting_tasks(model)


if __name__ == "__main__":
    evaluator = Evaluate()
    evaluator.evlaute_reflection_tests()

