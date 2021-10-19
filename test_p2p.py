from p2p.train import do_train

if __name__ == '__main__':
    do_train(client_num=50,
             num_neighbors=2,
             examples_range=(0, 90000000),
             batch_size=50,
             private_ds_size=-1,
             shared_pars={"v": 4, "lr": 0.005, "decay": 0, "default_weights": True},
             private_pars={"v": 1, "lr": 0.005, "decay": 0, "default_weights": True},
             seed=123,
             share_method='layer-average',
             epochs=60,
             accuracy_step='epoch',
             resume_agent_id=-1
             )

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
