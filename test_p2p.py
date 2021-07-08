from p2p.train import do_train

if __name__ == '__main__':
    do_train(client_num=20,
             num_neighbors=15,
             examples_range=(0, 900000),
             batch_size=50,
             complex_ds_size=-1,
             base_pars={"v": 2, "lr": 0.005, "decay": 0},
             complex_pars={"v": 2, "lr": 0.005, "decay": 0},
             seed=123,
             mode='RAM',
             share_method='average',
             epochs=10,
             resume_agent_id=-1
             )

# /Users/robert/.local/share/virtualenvs/p2p_sgd-oreLDV97/bin/python3.7 test_p2p.py > log/p2p_test.txt
# Training: 100%|██████████| 20/20 [00:55<00:00,  3.59s/it, RAM=9.12/16.0 GB, N_CPU=20]Training: 17 Useful: 300 Useless: 0
# Round: 1	Epoch: 1
# 	Val:	Mean: 0.439%	Median: 0.000%
# 		B:	Mean: 0.439%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 0.456%	Median: 0.000%
# 		B:	Mean: 0.456%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 0.855%	Median: 0.000%
# 		B:	Mean: 0.855%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:29<00:00,  1.25s/it, RAM=9.47/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 2	Epoch: 2
# 	Val:	Mean: 0.420%	Median: 0.000%
# 		B:	Mean: 0.420%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 0.760%	Median: 0.000%
# 		B:	Mean: 0.760%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 1.136%	Median: 0.234%
# 		B:	Mean: 1.136%	Median: 0.234%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:14<00:00,  1.16it/s, RAM=9.5/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 3	Epoch: 3
# 	Val:	Mean: 1.129%	Median: 0.000%
# 		B:	Mean: 1.129%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 1.419%	Median: 0.546%
# 		B:	Mean: 1.419%	Median: 0.546%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 2.203%	Median: 2.026%
# 		B:	Mean: 2.203%	Median: 2.026%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:15<00:00,  1.17s/it, RAM=9.47/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 4	Epoch: 4
# 	Val:	Mean: 2.254%	Median: 1.828%
# 		B:	Mean: 2.254%	Median: 1.828%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 3.439%	Median: 3.193%
# 		B:	Mean: 3.439%	Median: 3.193%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 4.070%	Median: 3.300%
# 		B:	Mean: 4.070%	Median: 3.300%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:11<00:00,  2.17it/s, RAM=9.52/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 5	Epoch: 5
# 	Val:	Mean: 2.453%	Median: 2.500%
# 		B:	Mean: 2.453%	Median: 2.500%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 4.313%	Median: 4.613%
# 		B:	Mean: 4.313%	Median: 4.613%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 5.825%	Median: 5.758%
# 		B:	Mean: 5.825%	Median: 5.758%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:11<00:00,  2.62it/s, RAM=9.5/16.0 GB, N_CPU=20]Training: 18 Useful: 300 Useless: 0
# Round: 6	Epoch: 6
# 	Val:	Mean: 2.797%	Median: 2.765%
# 		B:	Mean: 2.797%	Median: 2.765%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 4.551%	Median: 3.974%
# 		B:	Mean: 4.551%	Median: 3.974%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 7.588%	Median: 8.707%
# 		B:	Mean: 7.588%	Median: 8.707%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:11<00:00,  1.67it/s, RAM=9.61/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 7	Epoch: 7
# 	Val:	Mean: 4.736%	Median: 4.639%
# 		B:	Mean: 4.736%	Median: 4.639%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 4.633%	Median: 4.858%
# 		B:	Mean: 4.633%	Median: 4.858%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 10.527%	Median: 10.195%
# 		B:	Mean: 10.527%	Median: 10.195%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 20/20 [00:17<00:00,  1.02s/it, RAM=9.57/16.0 GB, N_CPU=20]Training: 19 Useful: 300 Useless: 0
# Round: 8	Epoch: 9
# 	Val:	Mean: 5.008%	Median: 5.623%
# 		B:	Mean: 5.008%	Median: 5.623%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 5.490%	Median: 5.375%
# 		B:	Mean: 5.490%	Median: 5.375%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 12.082%	Median: 12.815%
# 		B:	Mean: 12.082%	Median: 12.815%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training:  70%|███████   | 14/20 [00:07<00:02,  2.89it/s, RAM=9.49/16.0 GB, N_CPU=20]Total useful: 2400 Total useles's: 0
# Train agents: 3 minutes



#
#
#                                                             Init agents: 1 minutes
# Training 50 agents, num neighbors: 15, examples: 40238, share method: average
# Training: 100%|██████████| 50/50 [02:40<00:00,  6.96s/it]Training: 47 Useful: 750 Useless: 0
# Round: 1	Epoch: 0
# 	Val:	Mean: 0.930%	Median: 0.000%
# 		B:	Mean: 0.930%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 0.782%	Median: 0.000%
# 		B:	Mean: 0.782%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 1.281%	Median: 0.000%
# 		B:	Mean: 1.281%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:45<00:00,  3.78s/it]Training: 45 Useful: 750 Useless: 0
# Round: 2	Epoch: 2
# 	Val:	Mean: 0.801%	Median: 0.000%
# 		B:	Mean: 0.801%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 0.918%	Median: 0.000%
# 		B:	Mean: 0.918%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 1.435%	Median: 0.000%
# 		B:	Mean: 1.435%	Median: 0.000%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:14<00:00,  1.06it/s]Training: 46 Useful: 750 Useless: 0
# Round: 3	Epoch: 2
# 	Val:	Mean: 2.831%	Median: 1.719%
# 		B:	Mean: 2.831%	Median: 1.719%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 3.105%	Median: 2.681%
# 		B:	Mean: 3.105%	Median: 2.681%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 3.885%	Median: 4.554%
# 		B:	Mean: 3.885%	Median: 4.554%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:00<00:00,  1.41s/it]Training: 42 Useful: 750 Useless: 0
# Round: 4	Epoch: 3
# 	Val:	Mean: 3.832%	Median: 3.300%
# 		B:	Mean: 3.832%	Median: 3.300%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 3.803%	Median: 3.192%
# 		B:	Mean: 3.803%	Median: 3.192%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 5.954%	Median: 6.348%
# 		B:	Mean: 5.954%	Median: 6.348%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:11<00:00,  1.63s/it]Training: 48 Useful: 750 Useless: 0
# Round: 5	Epoch: 5
# 	Val:	Mean: 4.897%	Median: 4.048%
# 		B:	Mean: 4.897%	Median: 4.048%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 4.916%	Median: 4.832%
# 		B:	Mean: 4.916%	Median: 4.832%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 8.448%	Median: 8.462%
# 		B:	Mean: 8.448%	Median: 8.462%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [00:30<00:00,  1.64it/s]Training: 46 Useful: 750 Useless: 0
# Round: 6	Epoch: 5
# 	Val:	Mean: 6.060%	Median: 5.611%
# 		B:	Mean: 6.060%	Median: 5.611%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 5.313%	Median: 5.244%
# 		B:	Mean: 5.313%	Median: 5.244%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 9.344%	Median: 9.341%
# 		B:	Mean: 9.344%	Median: 9.341%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [00:39<00:00,  1.02s/it]Training: 48 Useful: 750 Useless: 0
# Training:   2%|▏         | 1/50 [00:00<00:00, 13148.29it/s]Round: 7	Epoch: 6
# 	Val:	Mean: 6.009%	Median: 5.611%
# 		B:	Mean: 6.009%	Median: 5.611%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 6.009%	Median: 5.652%
# 		B:	Mean: 6.009%	Median: 5.652%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 10.416%	Median: 10.267%
# 		B:	Mean: 10.416%	Median: 10.267%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:08<00:00,  1.27it/s]Training: 48 Useful: 750 Useless: 0
# Round: 8	Epoch: 7
# 	Val:	Mean: 6.274%	Median: 5.741%
# 		B:	Mean: 6.274%	Median: 5.741%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 6.583%	Median: 6.052%
# 		B:	Mean: 6.583%	Median: 6.052%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 11.379%	Median: 11.823%
# 		B:	Mean: 11.379%	Median: 11.823%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [00:41<00:00,  1.68it/s]Training: 46 Useful: 750 Useless: 0
# Training:   2%|▏         | 1/50 [00:00<00:00, 10645.44it/s]Round: 9	Epoch: 8
# 	Val:	Mean: 7.370%	Median: 7.378%
# 		B:	Mean: 7.370%	Median: 7.378%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 6.806%	Median: 6.202%
# 		B:	Mean: 6.806%	Median: 6.202%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 11.960%	Median: 12.123%
# 		B:	Mean: 11.960%	Median: 12.123%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [00:47<00:00,  1.15it/s]Training: 47 Useful: 750 Useless: 0
# Round: 10	Epoch: 8
# 	Val:	Mean: 7.432%	Median: 7.431%
# 		B:	Mean: 7.432%	Median: 7.431%
# 		C:	Mean: 0.000%	Median: 0.000%
# Training:   2%|▏         | 1/50 [00:00<00:00, 13189.64it/s]		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 7.260%	Median: 7.676%
# 		B:	Mean: 7.260%	Median: 7.676%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 12.555%	Median: 12.220%
# 		B:	Mean: 12.555%	Median: 12.220%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training: 100%|██████████| 50/50 [01:03<00:00,  1.72s/it]Training: 47 Useful: 750 Useless: 0
# Round: 11	Epoch: 9
# 	Val:	Mean: 7.459%	Median: 7.504%
# 		B:	Mean: 7.459%	Median: 7.504%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Test:	Mean: 6.571%	Median: 7.003%
# 		B:	Mean: 6.571%	Median: 7.003%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# 	Train:	Mean: 13.092%	Median: 13.633%
# 		B:	Mean: 13.092%	Median: 13.633%
# 		C:	Mean: 0.000%	Median: 0.000%
# 		E:	Mean: 0.000%	Median: 0.000%
# Training:  10%|█         | 5/50 [00:03<00:34,  1.32it/s]Total useful: 8250 Total useles's: 0
# Train agents: 13 minutes
