from main import MODEL_SETTINGS, DATASET_SETTINGS
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot(data_list, dataset_key, model_name):
    # Extract subject data from data_list
    subjects = data_list[0]['wrong'].keys()
    subjects_data = {}
    xs = []
    for s in subjects:
        s_data = []
        for data in data_list:
            n_correct = data['correct'][s]
            n_wrong = data['wrong'][s]
            s_data.append({'correct': n_correct, 'wrong': n_wrong})
        xs.append((s, [x['correct'] for x in s_data]))
        subjects_data[s] = s_data

    # Define plot parameters
    width = 0.1       # the width of the bars
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#5B84B1FF', '#5F4B8BFF', '#FC766AFF', '#E69A8DFF']   # list of RGB tuples defining shades of green
    edge_color = 'gray'    # color of the border
    legend_labels = MODEL_SETTINGS[model_name]['k-quants']    # labels for legend entries
    colors = colors[:len(legend_labels)]

    # Create dummy artists to use in the legend
    dummy_artists = [plt.Line2D([0], [0], color=c, lw=4) for c in colors]

    # Plot data
    for i,x in enumerate(xs):
        subj = x[0]
        values = x[1]
        for j, value in enumerate(values):  # for each value, we create a bar
            bar = plt.bar(i + j*width, value, width, color=colors[j % len(colors)])#, edgecolor=edge_color)   # use index plus an offset (depending on the value's position) as x-value and assign colors from list cyclically; add border around each bar
            plt.text(i + j*width - width/2, value + 0.1, str(value), color='black', fontweight='bold')   # place text above each bar with respective value

    # Add labels and legend to the plot
    plt.xticks(np.arange(len(xs)) + width/2, [x[0] for x in xs], rotation=40)  # set x-axis labels to subject texts with correct alignment and 40 degrees rotation
    plt.legend(dummy_artists, legend_labels)   # add legend with dummy artists and corresponding labels
    plt.title(dataset_key)
    plt.show()

    pass

def main():
    for dataset_key in DATASET_SETTINGS:
        print(dataset_key)
        for model_name in MODEL_SETTINGS:
            if model_name != 'mistral-7b':
                continue
            model = MODEL_SETTINGS[model_name]
            path = model['model_path']
            model_name_short = model_name
            model_name = model['model_name']
            quant_perf = []
            qs = MODEL_SETTINGS[model_name_short]['k-quants']
            for quant in MODEL_SETTINGS[model_name_short]['k-quants']:

                model_name_quant = model_name + f'-{quant}'
                
                dataset_name = dataset_key.replace('/', '-')
                path = f'results/{model_name}/{model_name_quant}/{model_name_quant}-{dataset_name}-results.json'
                
                if os.path.exists(path):
                    with open(path, 'r') as file:
                        data = json.load(file)
                        quant_perf.append(data)
                else:
                    print(f'Could not find: {path}')
            if len(quant_perf) > 0:
                plot(quant_perf, dataset_key, model_name_short)

if __name__ == '__main__':
    main()