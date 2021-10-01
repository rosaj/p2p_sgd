from p2p.train import do_train

if __name__ == '__main__':
    do_train(client_num=50,
             num_neighbors=5,
             examples_range=(0, 90000000),
             batch_size=50,
             complex_ds_size=-1,
             base_pars={"v": 3, "lr": 0.005, "decay": 0, "default_weights": True},
             complex_pars={"v": 3, "lr": 0.005, "decay": 0, "default_weights": False},
             seed=123,
             mode='RAM',
             share_method='layer-average',
             epochs=15,
             accuracy_step='10 iter',
             resume_agent_id=-1
             )

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
