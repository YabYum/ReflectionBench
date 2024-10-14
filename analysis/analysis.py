import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import beta
from scipy.optimize import minimize_scalar
import math
import seaborn as sns
from matplotlib.lines import Line2D
from ast import literal_eval
import re


def read_outcomes(models):
    
    results  = {}
    for model in models:
        file_path = f'../results/{model}_results.pkl'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        results[model] = data

    acc_2back = []
    dec1, dec2, rew1, rew2, rew_avg = [],[],[],[],[]
    rules1, tests1, options1, feedbacks1 = [],[],[],[]
    rules2, tests2, options2, feedbacks2 = [],[],[],[]
    metadec1, metadec2, metarew1, metarew2 = [],[],[],[]
    wpt_today1, wpt_device1, wpt_pred1, wpt_tomorrow1, wpt_today2, wpt_device2, wpt_pred2, wpt_tomorrow2 = [], [], [], [], [], [], [], []
    selections1_session1, gains1_session1, losss1_session1, overages1_session1, selections2_session1, gains_session1, losss_session1, overages_session1 = [],[],[],[],[],[],[],[]
    selections1_session2, gains1_session2, losss1_session2, overages1_session2, selections2_session2, gains_session2, losss_session2, overages_session2 = [],[],[],[],[],[],[],[]

    for model in models:
        #2-back
        acc1 = np.sum([item['is_correct'] for item in results[model]['2back']['outcomes'][0].values()])
        acc2 = np.sum([item['is_correct'] for item in results[model]['2back']['outcomes'][1].values()])
        mean_acc = (acc1 + acc2) / 104 * 100
        acc_2back.append(mean_acc)
        #prlt
        dec_0 = [item['decision'] for item in results[model]['prt']['outcomes'][0].values()]
        dec_1 = [item['decision'] for item in results[model]['prt']['outcomes'][1].values()]
        rew_0 = np.array([item['reward'] for item in results[model]['prt']['outcomes'][0].values()], dtype=float)
        rew_1 = np.array([item['reward'] for item in results[model]['prt']['outcomes'][1].values()], dtype=float)
        rew_00 = np.nansum(rew_0)
        rew_11 = np.nansum(rew_1)
        meanprt = (rew_00+rew_11)/80 * 100
        dec1.append(dec_0)
        dec2.append(dec_1)
        rew1.append(rew_0)
        rew2.append(rew_1)
        rew_avg.append(meanprt)
        #wcst
        rule1 = [item['rule'] for item in results[model]['wcst']['outcomes'][0].values()]
        rule2 = [item['rule'] for item in results[model]['wcst']['outcomes'][1].values()]
        test1 = [item['testing card'] for item in results[model]['wcst']['outcomes'][0].values()]
        test2 = [item['testing card'] for item in results[model]['wcst']['outcomes'][1].values()]
        option1 = [item['option'] for item in results[model]['wcst']['outcomes'][0].values()]
        option2 = [item['option'] for item in results[model]['wcst']['outcomes'][1].values()]
        feedback1 = [item['feedback'] for item in results[model]['wcst']['outcomes'][0].values()]
        feedback2 = [item['feedback'] for item in results[model]['wcst']['outcomes'][1].values()]
        rules1.append(rule1)
        rules2.append(rule2)
        tests1.append(test1)
        tests2.append(test2)
        options1.append(option1)
        options2.append(option2)
        feedbacks1.append(feedback1)
        feedbacks2.append(feedback2)
        #wpt    
        today0 = [item['today weather'] for item in results[model]['wpt']['outcomes'][0].values()]
        today1 = [item['today weather'] for item in results[model]['wpt']['outcomes'][1].values()]
        device0 = [item['device state'] for item in results[model]['wpt']['outcomes'][0].values()]
        device1 = [item['device state'] for item in results[model]['wpt']['outcomes'][1].values()]
        pred0 = [item['model prediction'] for item in results[model]['wpt']['outcomes'][0].values()]    
        pred1 = [item['model prediction'] for item in results[model]['wpt']['outcomes'][1].values()]
        pred0 = [pred.lower().strip().split()[-1].rstrip('.') for pred in pred0]
        pred1 = [pred.lower().strip().split()[-1].rstrip('.') for pred in pred1]
        tomorrow0 = [item['tomorrow weather'] for item in results[model]['wpt']['outcomes'][0].values()]
        tomorrow1 = [item['tomorrow weather'] for item in results[model]['wpt']['outcomes'][1].values()]
        wpt_today1.append(today0)
        wpt_today2.append(today1)
        wpt_device1.append(device0)
        wpt_device2.append(device1)
        wpt_pred1.append(pred0)
        wpt_pred2.append(pred1)
        wpt_tomorrow1.append(tomorrow0)
        wpt_tomorrow2.append(tomorrow1)
        # iowa gambling test
        selection0_session1 = [item['selection 1'] for item in results[model]['dcigt']['outcomes'][0].values()]
        gain0_session1 = [item['gain 1'] for item in results[model]['dcigt']['outcomes'][0].values()]
        loss0_session1 = [item['loss 1'] for item in results[model]['dcigt']['outcomes'][0].values()]
        overage0_session1 = [item['overage 1'] for item in results[model]['dcigt']['outcomes'][0].values()]
        selection1_session1 = [item['selection 2'] for item in results[model]['dcigt']['outcomes'][0].values()]
        gain1_session1 = [item['gain'] for item in results[model]['dcigt']['outcomes'][0].values()]
        loss1_session1 = [item['loss'] for item in results[model]['dcigt']['outcomes'][0].values()]
        overage1_session1 = [item['overage'] for item in results[model]['dcigt']['outcomes'][0].values()]

        selection0_session2 = [item['selection 1'] for item in results[model]['dcigt']['outcomes'][1].values()]
        gain0_session2 = [item['gain 1'] for item in results[model]['dcigt']['outcomes'][1].values()]
        loss0_session2 = [item['loss 1'] for item in results[model]['dcigt']['outcomes'][1].values()]
        overage0_session2 = [item['overage 1'] for item in results[model]['dcigt']['outcomes'][1].values()]
        selection1_session2 = [item['selection 2'] for item in results[model]['dcigt']['outcomes'][1].values()]
        gain1_session2 = [item['gain'] for item in results[model]['dcigt']['outcomes'][1].values()]
        loss1_session2 = [item['loss'] for item in results[model]['dcigt']['outcomes'][1].values()]
        overage1_session2 = [item['overage'] for item in results[model]['dcigt']['outcomes'][1].values()]

        selections1_session1.append(selection0_session1)
        gains1_session1.append(gain0_session1)
        losss1_session1.append(loss0_session1)
        overages1_session1.append(overage0_session1)
        selections2_session1.append(selection1_session1)
        gains_session1.append(gain1_session1)
        losss_session1.append(loss1_session1)
        overages_session1.append(overage1_session1)

        selections1_session2.append(selection0_session2)
        gains1_session2.append(gain0_session2)
        losss1_session2.append(loss0_session2)
        overages1_session2.append(overage0_session2)
        selections2_session2.append(selection1_session2)
        gains_session2.append(gain1_session2)
        losss_session2.append(loss1_session2)
        overages_session2.append(overage1_session2)
        #meta-prlt
        metadec_0 = [item['decision'] for item in results[model]['metaprt_interval4']['outcomes'][1].values()]
        metadec_1 = [item['decision'] for item in results[model]['metaprt_interval4']['outcomes'][2].values()]
        metarew_0 = np.array([item['reward'] for item in results[model]['metaprt_interval4']['outcomes'][1].values()])
        metarew_1 = np.array([item['reward'] for item in results[model]['metaprt_interval4']['outcomes'][2].values()])
        metadec1.append(metadec_0)
        metadec2.append(metadec_1)
        metarew1.append(metarew_0)
        metarew2.append(metarew_1)
    
    # 11th trail of llama-3.1-70b in 1st session DC-IGT meets an error, leading to all outcomes 'None', so we replace them with the outcomes of last trial (10th)
    overages1_session1[8][10] = overages1_session1[8][9]
    overages_session1[8][10] = overages_session1[8][9]
    selections1_session1[8][10] = selections1_session1[8][9]
    selections2_session1[8][10] = selections2_session1[8][9]
    losss1_session1[8][10] = losss1_session1[8][9]
    
    return [results, acc_2back, dec1, dec2, rew1, rew2, rew_avg, rules1, tests1, options1, feedbacks1,rules2, tests2, options2, feedbacks2, metadec1, metadec2, metarew1, metarew2, wpt_today1, wpt_device1, wpt_pred1, wpt_tomorrow1, wpt_today2, wpt_device2, wpt_pred2, wpt_tomorrow2, 
            selections1_session1, gains1_session1, losss1_session1, overages1_session1, selections2_session1, gains_session1, losss_session1, overages_session1, selections1_session2, gains1_session2, losss1_session2, overages1_session2, selections2_session2, gains_session2, losss_session2, overages_session2]


def oddball_scoring(filepath):
    models_odb =['o1-preview', 'o1-mini', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620', 'gemini-1.5-pro', 'llama-3.1-405b', 
        'llama-3.1-70b', 'llama-3.1-8b', 'qwen2.5-72b-instruct', 'qwen2.5-32b-instruct','qwen2.5-14b-instruct']

    tem_dict=dict()
    for key in models_odb:
        tem_dict[key]=0

    with open(filepath,'r',encoding='utf-8') as f:
        for line in f.readlines():
            data=json.loads(line)
            key = data["custom"]["resource"].split("_")[0]
            tem_dict[key]+=int(data["evaluation"]["conversation_evaluation"]["score"])

    y_label, y_mean=[], []
    for _,item in tem_dict.items():
        y_label.append(item)
        y_mean.append(item / 4.5)
    
    return y_mean,y_label


def plot_mmn_waveforms(amplitudes, model_names, time_range=(0, 7), peak_time=4.5, peak_sigma=0.5, noise_amplitude=0.75, noise_sigma=0.6, figsize=(10, 6)):
    
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    time = np.linspace(time_range[0], time_range[1], 1000)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#ed8687', '#feb29b', '#ffd19d', '#bed2c6', '#c9e0e5', '#85b8bb', '#c2c8d8', '#9b9ac1', '#cabad7', '#99bbe0', '#a7c4ba', '#fcccca', '#c6d59e']
    
    for amp, color in zip(amplitudes, colors):
        # Main peak
        wave = gaussian(time, -amp/4.5, peak_time, peak_sigma)
        
        # Add random Gaussian noise
        noise = np.random.normal(0, noise_amplitude, len(time))
        for i in range(len(time)):
            noise[i] += gaussian(time[i], noise_amplitude, np.random.uniform(time_range[0], time_range[1]), noise_sigma)
        
        wave += noise
        
        ax.plot(time, wave, color=color, alpha=0.8, linestyle='-')
    
    ax.set_title('Mismatch Negativity-like Wave')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(time_range)
    ax.grid(False)
    
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=model)
                       for color, model in zip(colors, model_names)]
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.8), loc='upper left', 
              fontsize=10)
    
    plt.tight_layout()
    plt.show()


def diagram_scores(array, ylabel, title, models):

    fig, ax = plt.subplots(figsize=(10, 6))    
    x = np.arange(len(models))
    bars = ax.bar(x, array, width=0.95, color=['#ed8687', '#feb29b', '#ffd19d', '#bed2c6', '#c9e0e5', '#85b8bb', '#c2c8d8', '#9b9ac1', '#cabad7', '#99bbe0', '#a7c4ba', '#fcccca', '#c6d59e'])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0,110)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def visualize_prlt_decision(dec1, dec2, ylabel, title, labels=None, target_decision='AAA', window_size=2, sigma=3, figsize=(12, 6)):
    plt.figure(figsize=figsize)

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    colors = ['#ed8687', '#feb29b', '#ffd19d', '#bed2c6', '#c9e0e5', '#85b8bb', '#c2c8d8', '#9b9ac1', '#cabad7', '#99bbe0', '#a7c4ba', '#fcccca', '#c6d59e']
    true_prob = np.concatenate([np.ones(20) * 0.9, np.ones(20) * 0.1])

    errors = []

    for i in range(len(dec1)):
        y1 = [1 if d == target_decision else 0 for d in dec1[i]]
        y2 = [1 if d == target_decision else 0 for d in dec2[i]]
        
        y_avg = [(a + b) / 2 for a, b in zip(y1, y2)]
        
        ratio = moving_average(y_avg, window_size)
        smoothed_ratio = gaussian_filter1d(ratio, sigma)
        x = range(len(smoothed_ratio))
        error = np.mean(np.abs(smoothed_ratio - true_prob[window_size-1:]))
        errors.append(error)

        color = colors[i % len(colors)]
        label = labels[i] if labels and i < len(labels) else f'Model {i+1}'
        plt.plot(x, smoothed_ratio, color=color, linewidth=2, label=label)
    plt.plot(range(40), true_prob, color='black', linestyle='--', linewidth=2, label='True Probability')

    plt.title(title)
    plt.xlabel('Trials')
    plt.ylabel(ylabel)
    plt.ylim(0, 1.1)
    plt.grid(False)
    plt.legend()
    plt.show()

    return errors


def log_weighted_average(x1, x2, epsilon=0.01):
    e = math.e
    w1 = 1 / math.log(x1 + epsilon + e)
    w2 = 1 / math.log(x2 + epsilon + e) 
    return (w1 * x1 + w2 * x2) / (w1 + w2)


def plot_accuracy_heatmap(shape_acc_1, color_acc_1, number_acc_1, 
                          shape_acc_2, color_acc_2, number_acc_2, model_name):
    data = np.array([
        shape_acc_1,
        color_acc_1,
        number_acc_1,
        shape_acc_2,
        color_acc_2,
        number_acc_2
    ])

    row_labels = ['Trial 1-18, Shape', 'Trial 19-32, Color', 'Trial 33-52, Number', 'Trial 53-72, Shape', 'Trial 73-90, Color', 'Trial 90-108, Number']
    col_labels = [f'{model_name[i]}' for i in range(13)]

    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=col_labels, yticklabels=row_labels, 
                cbar_kws={'label': 'Accuracy'})
    plt.title('Wisconsin Card Sorting Test, Accuracy by Trial Blocks')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def visualize_metaprt(data, session,model_name):

    data_transposed = data.T
    fig, ax = plt.subplots(figsize=(10, 6))
    custom_cmap = plt.cm.get_cmap('Greens')
    im = ax.imshow(data_transposed, cmap=custom_cmap, interpolation='nearest', aspect='auto')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom",fontsize=10)
    ax.set_title(f'Probability Reversal Task, p = 1, interval = 3, session{session}')
    ax.set_ylabel('Trials')
    ax.set_xticks(np.arange(data_transposed.shape[1]))
    ax.set_xticklabels(model_name,fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(1, len(model_name)):
        ax.axvline(x=i-0.5, color='white', linewidth=2)
    plt.tight_layout()
    plt.show()


def estimate_transition_probabilities(wpt_today, wpt_device, wpt_pred):

    transitions = [np.zeros((2, 2)) for _ in range(2)]
    counts = [np.zeros((2, 2)) for _ in range(2)]

    weather_map = {'sunny': 0, 'rainy': 1}

    for i in range(len(wpt_today) - 1):
        today = weather_map[wpt_today[i]]
        last_word = re.findall(r'\b\w+\b', wpt_pred[i])[-1]
        pred = weather_map[last_word]
        device = literal_eval(wpt_device[i])
        matrix_index = device.index(1)
        counts[matrix_index][today][pred] += 1

    for i in range(2):
        for j in range(2):
            row_sum = np.sum(counts[i][j])
            if row_sum > 0:
                transitions[i][j] = counts[i][j] / row_sum
            else:
                transitions[i][j] = [0.5, 0.5]

    return transitions


def visualize_wpt_heatmap(estimated_transitions, true_transitions):

    plt.figure(figsize=(10, 6))
    diffs = []

    for i in range(2):

        plt.subplot(2, 3, i*3 + 1)
        sns.heatmap(true_transitions[i], annot=True, cmap="Blues", vmin=0, vmax=1, annot_kws={"size": 10})
        plt.title(f"True Transition {i+1}", fontsize=10)
        plt.xlabel("Tomorrow", fontsize=10)
        plt.ylabel("Today", fontsize=10)
    
        plt.subplot(2, 3, i*3 + 2)
        sns.heatmap(estimated_transitions[i], annot=True, cmap="Blues", vmin=0, vmax=1, annot_kws={"size": 10})
        plt.title(f"Estimated Transition {i+1}", fontsize=10)
        plt.xlabel("Tomorrow", fontsize=10)
        plt.ylabel("Today", fontsize=10)
    
        plt.subplot(2, 3, i*3 + 3)
        diff = estimated_transitions[i] - true_transitions[i]
        diffs.append(diff)
        sns.heatmap(diff, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot_kws={"size": 10})
        plt.title(f"Difference (Estimated - True) {i+1}", fontsize=10)
        plt.xlabel("Tomorrow", fontsize=10)
        plt.ylabel("Today", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    return diffs


def score_diff(actual, diff):
    scores=[]      
    for k, transition in enumerate(actual):

        matrix = np.array(transition)
        max_error = 0
        for i in range(2):
            for j in range(2):
                error_to_0 = matrix[i, j]
                error_to_1 = 1 - matrix[i, j]
                max_error += max(error_to_0, error_to_1)

        score = (1- np.sum(np.abs(diff[k])) / max_error)*100
        scores.append(score)
    return log_weighted_average(scores[0],scores[1],epsilon=5)


def visualize_igt(overage_lists, title, labels, window_size=10, sigma=3, figsize=(12, 6)):

    plt.figure(figsize=figsize)

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    colors = ['#ed8687', '#feb29b', '#ffd19d', '#bed2c6', '#c9e0e5', '#85b8bb', '#c2c8d8', '#9b9ac1', '#cabad7', '#99bbe0', '#a7c4ba', '#fcccca', '#c6d59e']

    for i, overage in enumerate(overage_lists):
        y = overage
        ratio = moving_average(y, window_size)
        smoothed_ratio = gaussian_filter1d(ratio, sigma)
        x = range(len(smoothed_ratio))

        color = colors[i % len(colors)]
        label = labels[i]
        plt.plot(x, smoothed_ratio, color=color, linewidth=2, label=label)

    plt.title(title)
    plt.xlabel('Trials')
    plt.ylabel("Overage")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc=3)
    plt.show()


def beneficial_switching(selcetion1, selection2, loss1):
    switch_num = 0
    score = 0
    for i in range(100):
        if selcetion1[i] == selection2[i]:
            # insisting for expected gain
            if loss1[i] == 0:
                score += 1
            # insisting without avoiding risks
            elif loss1[i] > 0:
                score -= (loss1[i] / 260) + 1
        elif selcetion1[i] != selection2[i]:
            switch_num += 1
            if loss1[i] == 0:
                # switching unnecessarily
                score -= 0.1
            elif loss1[i] > 0:            
                # switching for less loss
                score += 1
    return switch_num, score


def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def overall(mean_oddball, acc_2back, prt_scores, wcst_score, scores, overall_scores_dcigt,model_name):

    tasks = ['Perception', 'Memory', 'Belief updating', 'Desicion making', 'Prediction', 'Counterfactual thinking']
    data = np.array([mean_oddball, acc_2back, prt_scores, wcst_score, scores, overall_scores_dcigt]).T
    colors = ['#ed8687', '#feb29b', '#ffd19d', '#bed2c6', '#c9e0e5', '#85b8bb', '#c2c8d8', '#9b9ac1', '#cabad7', '#99bbe0', '#a7c4ba', '#fcccca', '#c6d59e']
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    angles = np.linspace(0, 2*np.pi, len(tasks), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    for i, (model, color) in enumerate(zip(model_name, colors)):
        values = np.concatenate((data[i], [data[i][0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks, fontsize=13)
    ax.tick_params(axis='x', pad=25)
    plt.legend(loc=0, bbox_to_anchor=(1.3, 1.0))
    ax.set_ylim(0, 100)
    plt.title("Reflection-Bench", size=20, y=1.1)
    plt.tight_layout()
    plt.show()

    alls=[]
    for i in range(13):
        all_score = mean_oddball[i] + acc_2back[i] + prt_scores[i] + wcst_score[i] + scores[i] + overall_scores_dcigt[i]
        alls.append(all_score/6)
    diagram_scores(alls, 'Total score', 'Reflection-Bench', model_name)