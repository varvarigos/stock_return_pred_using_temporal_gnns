import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm

def run_notebook(notebook_path, parameters):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    param_cell = nbformat.v4.new_code_cell(source='\n'.join([f"{key} = {repr(value)}" for key, value in parameters.items()]))
    nb.cells.insert(0, param_cell)

    ep = ExecutePreprocessor(timeout=12000, kernel_name='python3')

    if 'exp_logs' in os.listdir():
        for file in os.listdir('exp_logs'):
            approach = parameters['approach']
            window = parameters['window']
            use_attention = parameters['use_attention']
            use_temporal = parameters['use_temporal']
            use_lstm_prediction = parameters['use_lstm_prediction']
            graph_weight_type = parameters['graph_weight_type']
            index = parameters['index']
            filename = f'approach_{approach}__window_{window}' + \
                (f'__use_attention' if use_attention else '') + \
                (f'__use_temporal' if use_temporal else '') + \
                (f'__use_lstm_prediction' if use_lstm_prediction else '') + \
                (f'__use_learnable_correlation' if graph_weight_type == 'learnable correlation' else '') + \
                (f'__run_index_{index}') + '.json'

            if file == filename:
                # print(f"File {filename} already exists in the directory. Skipping the execution.")
                return
    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        print(f"Error executing the notebook: {e}")
        raise

notebook_path = '/teamspace/studios/this_studio/GNNs project/coursework_code.ipynb'

approaches = [1, 2, 3]
windows = [7, 30, 180]
use_attentions = [True, False]
use_temporals = [False]
use_lstm_predictions = [False]
graph_weight_types = ['correlation', 'learnable correlation']
indices = [1, 2, 3]

# populate expperiment_1_params
experiment_1_params = []
for index in indices:
    for approach in approaches:
        for graph_weight_type in graph_weight_types:
            for window in windows:
                for use_attention in use_attentions:
                    experiment_1_params.append({
                        'approach': approach,
                        'window': window,
                        'use_attention': use_attention,
                        'use_temporal': False,
                        'use_lstm_prediction': False,
                        'graph_weight_type': graph_weight_type,
                        'index': index
                    })

# populate expperiment_2_params
experiment_2_params = []
for index in indices:
    for approach in approaches:
        for graph_weight_type in graph_weight_types:
            for use_attention in use_attentions:
                experiment_2_params.append({
                    'approach': approach,
                    'window': 20,
                    'use_attention': use_attention,
                    'use_temporal': True,
                    'use_lstm_prediction': False,
                    'graph_weight_type': graph_weight_type,
                    'index': index
                })

# populate expperiment_3_params
experiment_3_params = []
for index in indices:
    for approach in approaches:
        experiment_3_params.append({
            'approach': approach,
            'window': 20,
            'use_attention': False,
            'use_temporal': False,
            'use_lstm_prediction': True,
            'graph_weight_type': 'correlation',
            'index': index
        })

total_runs = len(experiment_1_params) + len(experiment_2_params) + len(experiment_3_params)

pbar = tqdm(total=total_runs)
for i, params in enumerate(experiment_1_params):
    run_notebook(notebook_path, params)
    pbar.update(1)
for i, params in enumerate(experiment_2_params):
    run_notebook(notebook_path, params)
    pbar.update(1)
for i, params in enumerate(experiment_3_params):
    run_notebook(notebook_path, params)
    pbar.update(1)

pbar.close()