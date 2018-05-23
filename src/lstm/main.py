import lstm
import data
import train
import params


if __name__ == '__main__':
    data_interface = data.Data()
    print data_interface.vocab_size           #  should be 2967
    training = train.Learn()

    for run in range(1, params.num_runs + 1):
        model = lstm.LSTM(run, data_interface.vocab_size)
        model.build_network()
        training.run_training(model, data_interface)
