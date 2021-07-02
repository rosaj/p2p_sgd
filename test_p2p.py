from p2p.train import do_train

if __name__ == '__main__':
    do_train(client_num=400,
             num_neighbors=15,
             examples_range=(0, 900000),
             batch_size=50,
             complex_ds_size=3_000,
             base_pars={"v": 2, "lr": 0.005, "decay": 0},
             complex_pars={"v": 2, "lr": 0.005, "decay": 0},
             seed=123,
             mode='DISK',
             share_method='average',
             epochs=30,
             resume_agent_id=-1
             )

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
