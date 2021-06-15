import networkx as nx
import numpy as np
import warnings
import h5py
from keras import Model
import random
#import matplotlib.pyplot as plt


def load_optimizer_weights(model: Model, model_save_path: str):
    """
    Loads optimizer's weights for the model from an HDF5 file.
    """
    with h5py.File(model_save_path, mode='r') as f:
        if 'optimizer_weights' in f:
            # Build train function (to get weight updates).
            # noinspection PyProtectedMember
            model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [
                n.decode('utf8') for n in
                optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [
                optimizer_weights_group[n]
                for n in optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')


def contain_tf_gpu_mem_usage():
    """
    By default TensorFlow may try to reserve all available GPU memory
    making it impossible to train multiple networks at once.
    This function will disable such behaviour in TensorFlow.
    """
    from keras import backend
    if backend.backend() != 'tensorflow':
        return
    try:
        # noinspection PyPackageRequirements
        import tensorflow as tf
    except ImportError:
        pass
    else:
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory
        sess = tf.Session(config=config)
        set_session(sess)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def expand_vectors(vector):
    # vector is a one-hot vector
    N = len(vector)
    expanded = [0 for i in range(N * N)]
    vectors_class = -1
    for i in range(N):
        if vector[i] == 1:
            vectors_class = i
    if vectors_class == -1:
        return np.array(expanded)
    else:
        expanded[vectors_class * (N + 1)] = 1
        return np.array(expanded)


def expand_arrays(Y):
    Y_2 = []
    for i in range(len(Y)):
        Y_2.append(expand_vectors(Y[i]))
    return np.array(Y_2)


# def plot_graph(G, labels):
#     nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), color=labels)
#     plt.show()


def load_data_anomalies(G, anomaly_param):
    """Creates dataset with anomalies"""
    number_of_attributes = 5
    number_of_nodes = G.number_of_nodes()

    number_of_communities = 2  # anomaly and non-anomaly
    adj = nx.to_numpy_array(G)
    features = np.zeros((number_of_nodes, number_of_attributes))

    for i in range(number_of_nodes):
        random.seed(a=i)  # for reproducible results
        value = random.random()
        for j in range(number_of_attributes):
            if (1 / number_of_attributes) * j < value < (1 / number_of_attributes) * (j + 1):
                features[i][j] = 1

    # bleu =  [1, 0, 0, 0, 0]
    # vert =  [0, 1, 0, 0, 0]
    # rouge = [0, 0, 1, 0, 0]
    # blanc = [0, 0, 0, 1, 0]
    # noir =  [0, 0, 0, 0, 1]

    anomalies_list = []
    number_of_anomalies = len(anomalies_list)
    boucle = 0
    minimum_pourcentage_of_anomalies = 0.04
    maximum_pourcentage_of_anomalies = 0.06

    while not int(minimum_pourcentage_of_anomalies * number_of_nodes) < number_of_anomalies < int(maximum_pourcentage_of_anomalies * number_of_nodes):
        anomalies_list = []
        boucle += 1
        for i in range(number_of_nodes):
            bleu = 0
            vert = 0
            rouge = 0
            blanc = 0
            noir = 0
            self_bleu = 0
            self_vert = 0
            self_rouge = 0
            self_blanc = 0
            self_noir = 0
            if features[i][0] == 1:
                self_bleu = 1
            elif features[i][1] == 1:
                self_vert = 1
            elif features[i][2] == 1:
                self_rouge = 1
            elif features[i][3] == 1:
                self_blanc = 1
            elif features[i][4] == 1:
                self_noir = 1
            liste = list(G[i])
            for j in range(len(liste)):
                if features[liste[j]][0] == 1:  # bleu
                    bleu = 1
                elif features[liste[j]][1] == 1:  # vert
                    vert = 1
                elif features[liste[j]][2] == 1:  # rouge
                    rouge = 1
                elif features[liste[j]][3] == 1:  # blanc
                    blanc = 1
                elif features[liste[j]][4] == 1:  # noir
                    noir = 1
            if anomaly_param == 0:
                if (bleu and vert):
                    anomalies_list.append(i)
            elif anomaly_param == 1:
                if (bleu and vert) or (bleu and rouge):
                    anomalies_list.append(i)
            elif anomaly_param == 2:
                if (bleu and vert) or (rouge and noir):
                    anomalies_list.append(i)
            elif anomaly_param == 3:
                if (self_noir):
                    anomalies_list.append(i)
            elif anomaly_param == 4:
                if (self_noir and bleu):
                    anomalies_list.append(i)
            elif anomaly_param == 5:
                if (self_noir and bleu and noir):
                    anomalies_list.append(i)
            elif anomaly_param == 6:
                if (self_noir) or (bleu and noir):
                    anomalies_list.append(i)
                    
        number_of_anomalies = len(anomalies_list)
        if not int(minimum_pourcentage_of_anomalies * number_of_nodes) < number_of_anomalies < int(maximum_pourcentage_of_anomalies * number_of_nodes):
            if number_of_anomalies < int(minimum_pourcentage_of_anomalies * number_of_nodes):
                raise RuntimeError
            for i in range(number_of_nodes):
                value = random.random()
                if value < 0.005:
                    for j in range(len(features[0])):
                        features[i][j] = 0
                    features[i][3] = 1   # change the color of the node to white
    normal_list = []
    for i in range(number_of_nodes):
        if i not in anomalies_list:  # the other nodes are normal
            normal_list.append(i)

    labels = [[0, 1] for i in range(number_of_nodes)]  # labels of normal nodes
    for i in anomalies_list:
        labels[i] = [1, 0]  # labels of anomalies
    labels = np.array(labels)
    features = np.array(features)

    random.seed()  # change the seed of random to change sampling
    anomalies_list = random.sample(anomalies_list, len(anomalies_list))  # shuffle the list of anomalies
    normal_list = random.sample(normal_list, len(normal_list))  # shuffle the list of normal nodes

    c = len(anomalies_list)
    d = len(normal_list)

    idx_test = list(anomalies_list[int(0.75 * c):]) + list(normal_list[int(0.75 * d):])
    d = c  # re-equilibrate train and validation
    idx_train = list(anomalies_list[:int(0.5 * c)]) + list(normal_list[:int(0.5 * d)])
    idx_val = list(anomalies_list[int(0.5 * c):int(0.75 * c)]) + list(normal_list[int(0.5 * d):int(0.75 * d)])

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print('Number of nodes: ')
    print(G.number_of_nodes())
    print('Number of edges: ')
    print(G.number_of_edges())
    print('Number of anomalies: ')
    print(len(anomalies_list))

    ############################################
    ## plot degree distribution of anomalies and normal nodes
    ############################################

    # degree_list = G.degree()
    # degree_list_anomalies = [d for n, d in degree_list if n in anomalies_list]
    # degree_list_all = [d for n, d in degree_list]
    # max_degree = max(degree_list_all)+1
    # hist_anomalies = [0 for i in range(max_degree)]
    # hist_all = [0 for i in range(max_degree)]
    # for d in degree_list_anomalies:
    #     hist_anomalies[d] += 1
    # for d in degree_list_all:
    #     hist_all[d] += 1
    # hist = [hist_anomalies[d]/hist_all[d] if hist_all[d] != 0 else 0 for d in range(max_degree)]
    # degrees = range(max_degree)
    #
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('Degree')
    # ax1.set_ylabel('Number of nodes', color=color)
    # ax1.plot(degrees, hist_all, 'r+-')
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()
    # color = 'tab:blue'
    # ax2.set_ylabel('Number of anomalies', color=color)
    # ax2.plot(degrees, hist_anomalies, 'b+-')
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()
    #
    # title = 'A' + str(anomaly_param) + '_' + G.name
    # plt.title(title)
    # plt.savefig('./figure_distribution_anomalies/' + title)
    # # plt.show()

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels, anomalies_list, normal_list
