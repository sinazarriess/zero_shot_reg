import lstm
import data
import train
import params
import numpy as np
import json
import params as p

def generate_indextotoken(data):
    with open(p.results_data_dir + '/raw_dataset_filenames.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['filenames'], delimiter=', ', fmt="%s")

    with open(p.results_data_dir + '/index2token.json', 'w') as f:
        json.dump(data.index_to_token, f)

    with open(p.results_data_dir + '/raw_dataset.txt', 'w') as f:
        np.savetxt(f, data.raw_dataset['test']['images'], delimiter=', ')


if __name__ == '__main__':

    words_excluded = ["juice", "toy", "kitty", "circle", "soldier"]
    data_interface = data.Data(words_excluded)
    print data_interface.vocab_size           #  should be 2967
    training = train.Learn()

    for run in range(1, params.num_runs + 1):
        model = lstm.LSTM(run, data_interface.vocab_size)
        model.build_network()
        training.run_training(model, data_interface)

    generate_indextotoken(data_interface)

