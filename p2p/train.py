from p2p.p2p_training import train_loop


def do_train(agent_class,
             clients_data_pars,
             model_pars=None,
             agent_pars=None,
             graph_pars=None,
             epochs=40,
             seed=None,
             accuracy_step='epoch'
             ):

    train_loop(
        agent_class=agent_class,
        clients_data_pars=clients_data_pars,
        model_pars=model_pars,
        graph_pars=graph_pars,
        agent_pars=agent_pars,
        epochs=epochs, seed=seed, accuracy_step=accuracy_step)
