from fl import do_train

if __name__ == '__main__':
    do_train(client_num=100,
             num_train_clients=5,
             examples_range=(0, 900000),
             batch_size=50,
             epochs=80,
             client_pars={"lr": 0.005, "decay": 0},
             server_pars={"lr": 0.005, "decay": 0},
             model_v=2,
             client_weighting='num_examples',
             round_num=None,
             seed=123,
             accuracy_step='epoch'
             )

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_fl.py > log/fl_50_1000.txt
