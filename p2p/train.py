from p2p.p2p_training import train_loop


def do_train(agent_pars,
             agent_data_pars,
             model_pars,
             graph_pars,
             sim_pars
             ):

    assert all(k in agent_pars for k in ["agent_class"])
    assert all(k in agent_data_pars for k in ["agents_data", "num_clients", "batch_size"])
    assert all(k in graph_pars for k in ["graph_type", "num_neighbors", "directed"])

    train_loop(agent_pars=agent_pars,
               agent_data_pars=agent_data_pars,
               model_pars=model_pars,
               graph_pars=graph_pars,
               sim_pars=sim_pars)
