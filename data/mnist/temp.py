"""
  # train, val, test = clients_data.load_clients_data(num_clients, starting_client)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    ctr, cte = [], []
    for _ in range(num_clients):
        train_ind = np.random.randint(len(x_train), size=int(len(x_train)/num_clients))
        test_ind = np.random.randint(len(x_test), size=int(len(x_test)/num_clients))
        ctr.append((x_train[train_ind], y_train[train_ind]))
        cte.append((x_test[test_ind], y_test[test_ind]))

    # train = np.array(ctr)
    # test = np.array(ctr)
    train = ctr
    test = cte
    val = train
"""
