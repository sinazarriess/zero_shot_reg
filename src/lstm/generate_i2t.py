import data
import params as p
import json
import numpy as np

if __name__ == '__main__':
    data_interface = data.Data()
    with open(p.results_data_dir + '/index2token.json', 'w') as f:
        json.dump(data_interface.index_to_token, f)

    with open(p.results_data_dir + '/raw_dataset_filenames.txt', 'w') as f:
        np.savetxt(f, data_interface.raw_dataset['test']['filenames'], delimiter=', ', fmt="%s")


    with open(p.results_data_dir + '/raw_dataset.txt', 'w') as f:
        np.savetxt(f, data_interface.raw_dataset['test']['images'], delimiter=', ')

    with open(p.results_data_dir + '/raw_dataset_filenames_train.txt', 'w') as f:
        np.savetxt(f, data_interface.raw_dataset['train']['filenames'], delimiter=', ', fmt="%s")

    with open(p.results_data_dir + '/raw_dataset_train.txt', 'w') as f:
        np.savetxt(f, data_interface.raw_dataset['train']['images'], delimiter=', ')