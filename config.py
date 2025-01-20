# config.py

CONFIG = {
    'api_setting':{

        # Setting API for evaluated model
        'base_url' : "https://api.claudeshop.top/v1",
        'api_key' : "sk-W8y2S1Op0NjLvQ17G313msRgnHjRWDDW1MeVBqGwNIcauFSQ",

        # Setting API for extractor. We use DeepSeek (https://api.deepseek.com) for extracting results from model's responses
        'base_url_dpsk': "https://api.deepseek.com",
        'api_key_dpsk': "sk-8c0b70e1951044c98afab5b13481ceb3",        

        # Setting API for automatically scoring Oddball Test 
        'api_key_embedding' :"",
        'base_url_embedding' : "",
    },

    'oddball': {
        'sessions': 3
    },


    'nback': {
        'sessions': 2,
        'back_n': 2 # if you want to change the 'n' of n-back task, remember also replace the n of system prompts (find it at \dataset\systemprompts.json)
    },

    'prlt': {
        'trials': 40,
        'p': 0.8,
        'sessions': 2
    },

    'wcst': {
        'rule_change_interval': 12,
        'sessions': 2
    },

    'wpt': {
        'sessions': 2,
        'transition_1': [
            [0.9 , 0.1],
            [0.1 , 0.9]
        ],      
        'transition_2' : [
            [0.1, 0.9], 
            [0.9, 0.1]
            ],
        'estimate_interval': 20
    },

    'dcigt': {
        'trials': 50,
        'sessions': 2,
        'p_loss' : {'AAA':0.5, 'BBB':0.1, 'CCC':0.5, 'DDD':0.1},
        'num_loss' : {'AAA':260, 'BBB':1250, 'CCC':50, 'DDD':200},
        'num_gain' : {'AAA':100, 'BBB':100, 'CCC':50, 'DDD':50}
    },

    # we originally degisned a rule-change pattern for MBT, which means the reward reverses every interval_1 steps for the first half trials and every interval_2 steps
    # but we found that single changing pattern is already challenching enough for all of current models.
    # so when you want to change the parameters, make sure that {trials}/2 % {interval_1} = 0 & {trials}/2 % {interval_2} = 0.
    # when interval_1 != interval_2, their minimal common multiple k should satisfy {trials}/2 % k = 0
    # for example, if you want to change the rewards every 3 trials, you should set trials as 2 * 3 * m, where m is a positive integral.
    # when interval_1 = 2 and interval_2 = 3, trials should be 2 * (2*3) * m, where m is a positive integral.

    'metaprt': {
        'trials': 40,
        'p': 1,
        'sessions': 2,
        'interval_1': 2,
        'interval_2': 2
    }
}