from p2p import p2p_training
from data import clients_data

seed = 13
p2p_training.tf.random.set_seed(seed)
p2p_training.np.random.seed(seed)
train_clients, val_clients, test_clients = clients_data.load_client_datasets(1_000)
train_cli, val_cli, test_cli = p2p_training.filter_clients(train_clients, val_clients, test_clients, 1, (0, 100000000))
agents = p2p_training.init_agents(train_cli, val_cli, test_cli, 50, -1,
                                  base_pars={"v": 2, "lr": 0.005, "decay": 0}, complex_pars={"v": 2, "lr": 0.005, "decay": 0})


a1 = agents[0]
a1.fit()
a1.base_model.save_weights('test_weights.h5')

"""
train_cli = data.train_clients
lens, co = [len(train_cli[i][1]) for i in range(len(train_cli))], 0
mx = 0
import matplotlib.pyplot as plt
plt.hist(lens)
for i, l in enumerate(lens):
    if mx < l:
        print(i, l)
        mx = l
    if l > 300:
        co += 1
print(co)

a1 = agents[0]
a2 = agents[13]
a1.fit()
a2.fit()
a1._train_rounds = 1
a2._train_rounds = 1

# a2.train_fn = None
a2.receive_model(a1.get_share_weights(), mode='test', only_improvement=False)
a2._calc_new_accs()
a2.hist_acc
train.print_all_accs([a2], 1, breakdown=False)


a = agents[0]
a.train_len
a.base_model = train.load("teacher_small.h5")
# a._calc_new_accs()
for _ in range(5):
    a._train_rounds = 1
    a.fit()
    train.print_all_accs(agents, 1)



for i in range(20):
    agents[i]._train_rounds = 1
    agents[i].fit()

a = agents[41]
a.fit()
for i in range(20):
    # a.receive_update(agents[i].base_model.trainable_variables)
    a.base_model.set_weights(agents[i].base_model.get_weights())
    a._calc_new_accs()
aw = train.average_weights([agents[i].base_model.get_weights() for i in range(20)])
a.base_model.set_weights(aw)
a._calc_new_accs()


a = agents[13]
a.train_len
a._train_rounds = 1
a.fit()
a.hist_acc
w = []
a_preds = a.base_model(a.train_x[:5])
for i in range(20):
    a_j = agents[i]
    a_j._train_rounds = 1
    # a_j.fit()
    b_preds = a_j.base_model(a.train_x[:5])
    print(i, a.kl_loss(a_preds, b_preds).numpy())
    w.append(a_j.get_share_weights())

# w.append(a.get_share_weights())
# w = [a.get_share_weights(), train.average_weights(w)]
w = [a.get_share_weights(), agents[3].get_share_weights()]
a.base_model.set_weights(train.average_weights(w))
# a.base_model.set_weights(w[0])
a._calc_new_accs()
a.hist_acc

a_preds = a.base_model(a.train_x)
# b_preds = a.base_model(a.train_x)
b_preds = agents[1].base_model(a.train_x)

a.kl_loss(a_preds, b_preds).numpy()


a1.train_len
a2.train_len
w = a1.get_share_weights()
# w * 0.14338085539714868
for i in range(len(w)):
    w[i] = w[i] * 0.14338085539714868
    
"""
