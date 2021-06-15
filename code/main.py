import argparse
import os
from keras import optimizers, losses
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
import random
import networkx as nx
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC

from .utils import load_optimizer_weights, contain_tf_gpu_mem_usage, expand_arrays, load_data_anomalies
from .models import (graph_model)

# uncomment to draw decision trees
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus


def main(model_save_path: str,
         G,
         num_epochs: int,
         learning_rate: float,
         load_weights_only: bool,
         self_loops: bool,
         algo: int,
         anomaly_param: int
         ):
    """Main function"""
    contain_tf_gpu_mem_usage()

    ###################################################
    # name the anomalies
    ###################################################

    graph_name = G.name
    if anomaly_param == 0:
        anomaly_type = '(bleu and vert)'
    elif anomaly_param == 1:
        anomaly_type = '(bleu and vert) or (bleu and rouge)'
    elif anomaly_param == 2:
        anomaly_type = '(bleu and vert) or (rouge and noir)'
    elif anomaly_param == 3:
        anomaly_type = '(self_noir)'
    elif anomaly_param == 4:
        anomaly_type = '(self_noir and bleu)'
    elif anomaly_param == 5:
        anomaly_type = '(self_noir and bleu and noir)'
    elif anomaly_param == 6:
        anomaly_type = '(self_noir) or (bleu and noir)'
    else:
        print('No anomaly type, add option --anomaly_param #')
        return 0

    # add self-loops in the name if necessary
    if self_loops == 'True' or self_loops == True:
        graph_name += '_with_self_loops'
        anomaly_type += '_with_self_loops'
    else:
        anomaly_type += 'no_loops'

    # add name of the algo
    if algo == 0:
        anomaly_type += '_CoBaGAD'
    else:
        print('Algo number out. You must specify a number between 0 and 4')
        return 0
    print(anomaly_type)

    # choose dataset to load
    # A: adjacency matrix   N x N
    # X: feature matrix   N x F
    # labels: one-hot labels for every node   N x n_comm
    # Y_train: labels but vectors are zeros if nodes not in training set   N x n_comm
    # idx_train: vector, 1 if node i is in training set, 0 otherwise   N
    A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test, labels, anomalies_list, normal_list = load_data_anomalies(G, anomaly_param)

    N = X.shape[0]  # number of nodes
    F = X.shape[1]  # number of features
    n_comm = len(Y_train[0])  # number of classes
    batch_size = N  # set batch_size to number of nodes

    Y_train, Y_val, Y_test, labels = expand_arrays(Y_train), expand_arrays(Y_val), expand_arrays(Y_test), expand_arrays(labels)  # expand size of labels

    if self_loops == 'True' or self_loops == True:
        A = A + np.eye(A.shape[0])  # add self-loops to adjacency matrix
    else:
        pass

    def compile_new_model():
        optimizer = optimizers.Adam(lr=learning_rate)

        _model = graph_model(algo=algo, F=F, N=N, f_=n_comm)
            
        _model.compile(optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])

        return _model

    # load weights if they already exist
    if os.path.exists(model_save_path):
        if load_weights_only:
            print('Loading weights from', model_save_path)
            model = compile_new_model()
            model.load_weights(model_save_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_save_path)
        else:
            pass
    else:
        model = compile_new_model()

    model.summary()

    # set monitoring of the learning
    es_callback = EarlyStopping(monitor='val_loss', patience=1000)
    tb_callback = TensorBoard(batch_size=batch_size)
    mc_callback = ModelCheckpoint('A_' + str(anomaly_param) + "_" + graph_name + ".h5",
                                  monitor='val_weighted_acc',
                                  save_best_only=True,
                                  save_weights_only=True)

    # validation set: feature matrix, adjacency matrix, labels of the validation set, index of validation set
    validation_data = ([X, A], Y_val, idx_val)

    history = model.fit([X, A],
                        Y_train,
                        sample_weight=idx_train,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        shuffle=False,  # Shuffling data means shuffling the whole graph
                        callbacks=[es_callback, tb_callback, mc_callback],
                        verbose=1)

    ##################################################################################
    # Doing test
    ##################################################################################

    model.load_weights(model_save_path)

    c = len(anomalies_list)
    d = len(normal_list)

    y_pred = model.predict([X, A], batch_size=batch_size)
    number_of_anomalies = len(anomalies_list)
    y_pred_test = np.vstack((y_pred[anomalies_list[int(0.75 * c):]], y_pred[normal_list[int(0.75 * d):]]))
    labels_test = np.vstack((labels[anomalies_list[int(0.75 * c):]], labels[normal_list[int(0.75 * d):]]))

    print("confusion matrix for test set")
    Az = confusion_matrix(y_pred_test.argmax(axis=1) % 2, labels_test.argmax(axis=1) % 2)
    print(Az)

    ##################################################################################
    # generating local explanations
    ##################################################################################

    start = int(0.75 * len(anomalies_list))
    explain_set = anomalies_list[start:]

    def X_modif(G, X_initial, id, reroll_pourcentage, blank_vector):
        X_bis = np.copy(X_initial)
        neighbors_of_id = list(G.neighbors(id))
        neighbors_of_id.append(id)
        for j in neighbors_of_id:
            if random.random() < reroll_pourcentage:
                X_bis[j] = blank_vector
        return X_bis

    iteration = 0
    for id in explain_set:
        iteration += 1
        progress = iteration / len(explain_set)
        print("Progress: " + str(progress))
        # blank_vector = [0.2, 0.2, 0.2, 0.2, 0.2]
        blank_vector = [0., 0., 0., 0., 0.]
        reroll_pourcentage = 0.1
        num_iterations = 250
        X_initial = np.copy(X)

        Xs = []  # features of the nodes (perturbed or not)
        class_s = []  # predicted class of nodes (with or without perturbation)

        X_0 = np.copy(X_initial)
        y_0 = model.predict([X_0, A], batch_size=batch_size)
        class_0 = np.argmax(y_0[id]) % 2

        if class_0 != 0:
            print('continue')
            continue

        neighbors_of_id = list(G.neighbors(id))
        neighbors_of_id.append(id)

        number_of_neighbors = len(neighbors_of_id)
        X_0 = X_0[neighbors_of_id]
        X_0 = X_0.flatten()

        Xs.append(X_0)
        class_s.append(class_0)

        for i in range(num_iterations):
            X_i = X_modif(G, X_initial, id, reroll_pourcentage, blank_vector)
            y_i = model.predict([X_i, A], batch_size=batch_size)
            class_i = np.argmax(y_i[id]) % 2
            X_i = X_i[neighbors_of_id]
            X_i = X_i.flatten()
            Xs.append(X_i)
            class_s.append(class_i)

        class_s = np.array(class_s)
        Xs = np.array(Xs)

        dtc = DTC(random_state=0, max_depth=3)
        dtc.fit(Xs, class_s)
        importance = dtc.feature_importances_
        importance2 = dtc.feature_importances_
        importance_type = 'best'  # 'best' for highest importance, 'all' for all non-zero
        predicted_features = np.array([0 for i in range(number_of_neighbors * 5)])
        predicted_features2 = np.array([0 for i in range(number_of_neighbors * 5)])
        tree_pred = dtc.predict(X_0.reshape(-1, number_of_neighbors * 5))
        ground_truth = []
        ground_truth2 = []

        # uncomment to draw decision tree

        # dot_data = StringIO()
        # export_graphviz(dtc, out_file=dot_data,
        #                 filled=True, rounded=True,
        #                 special_characters=True, class_names=['0', '1'])
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # image_title = './decision_tree' + str(iteration) + '.png'
        # graph.write_png(image_title)
        # Image(graph.create_png())

        # colors : 0 bleu, 1 vert, 2 rouge, 3 blanc, 4 noir

        if anomaly_param == 0:  # '(bleu and vert)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5): # for all neighbors
                if k % 5 == 2 or k % 5 == 3 or k % 5 == 4:
                    ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5): # for node itself
                ground_truth[k] = 0
            #  just to save results easily
            ground_truth2 = ground_truth

        elif anomaly_param == 1:  # '(bleu and vert) or (bleu and rouge)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 2 or k % 5 == 3 or k % 5 == 4:
                    ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                ground_truth[k] = 0
            ground_truth2 = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 1 or k % 5 == 3 or k % 5 == 4:
                    ground_truth2[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                ground_truth2[k] = 0

        elif anomaly_param == 2:  # '(bleu and vert) or (rouge and noir)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 2 or k % 5 == 3 or k % 5 == 4:
                    ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                ground_truth[k] = 0
            ground_truth2 = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 1 or k % 5 == 3 or k % 5 == 0:
                    ground_truth2[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                ground_truth2[k] = 0

        elif anomaly_param == 3:  # '(self_noir)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                if k % 5 != 4:
                    ground_truth[k] = 0
            ground_truth2 = ground_truth

        elif anomaly_param == 4:  # '(self_noir and bleu)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 1 or k % 5 == 2 or k % 5 == 3 or k % 5 == 4:
                    ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                if k % 5 != 4:
                    ground_truth[k] = 0
            ground_truth2 = ground_truth

        elif anomaly_param == 5:  # '(self_noir and bleu and noir)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 1 or k % 5 == 2 or k % 5 == 3:
                    ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                if k % 5 != 4:
                    ground_truth[k] = 0
            ground_truth2 = ground_truth

        elif anomaly_param == 6:  # '(self_noir) or (bleu and noir)'
            ground_truth = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                ground_truth[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                if k % 5 != 4:
                    ground_truth[k] = 0
            ground_truth2 = np.copy(X_0)  # node itself is last
            for k in range((number_of_neighbors - 1) * 5):  # for all neighbors
                if k % 5 == 1 or k % 5 == 2 or k % 5 == 3:
                    ground_truth2[k] = 0
            for k in range((number_of_neighbors - 1) * 5, number_of_neighbors * 5):  # for node itself
                ground_truth2[k] = 0

        else:
            print('No defined anomaly type, add option --anomaly_param #')
            return 0

        ################################################
        # write results in files
        ################################################

        graph_name = G.name
        result_file = "./results/result_" + str(graph_name) + "_A_" + str(anomaly_param) + ".txt"
        predicted_features = importance

        # create files to save results
        if not os.path.exists(result_file):
            with open(result_file, 'w'):
                pass

        with open(result_file, 'a') as f:
            string = "ground_truth1: "
            for i in range(len(ground_truth)):
                string += str(ground_truth[i]) + " "
            string += '\n'
            string += "ground_truth2: "
            for i in range(len(ground_truth2)):
                string += str(ground_truth2[i]) + " "
            string += '\n'
            string += "feature_pred1: "
            for i in range(len(predicted_features)):
                string += str(predicted_features[i]) + " "
            string += '\n'
            string += '\n'

            f.write(string)

    return 0


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='An example model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument(
        '--save', type=str, required=True, metavar='PATH',
        help='A path to save the best model')
    _argparser.add_argument(
        '--epochs', type=int, default=1000, metavar='INTEGER',
        help='The number of epochs to train')
    _argparser.add_argument(
        '--lr', type=float, default=5e-3, metavar='FLOAT',
        help='Learning rate')
    _argparser.add_argument(
        '--load-weights-only', action='store_true', default='True',
        help='Use the save file only to initialize weights ')
    _argparser.add_argument(
        '--loops', action='store_true', default='False')
    _argparser.add_argument(
        '--algo', type=int, default=0, metavar='INTEGER')
    _argparser.add_argument(
        '--anomaly_param', type=int, default=0, metavar='INTERGER',
        help='Type of anomaly')
    _args = _argparser.parse_args()
    print(_args)

    graph_number = 0  # start
    number_of_graphs = 6    # total number of graphs
    number_of_times_a_graph_is_computed = 1   # number of computations per graph
    max_graph_number = number_of_graphs * number_of_times_a_graph_is_computed    # total number of computations
    while graph_number < max_graph_number:
        # if os.path.exists(_args.save):
        #     os.remove(_args.save)
        if graph_number % number_of_graphs == 0:
            G = nx.generators.gnp_random_graph(n=10000, p=0.0005, seed=42)
            G.name = 'Erdos-Renyi_10000'
        elif graph_number % number_of_graphs == 1:
            file = './code2/data/dancer10000/dancer10000.edgeList'
            G = nx.read_edgelist(file, nodetype=int)
            G.name = 'dancer_10000'
        elif graph_number % number_of_graphs == 2:
            file = './code2/data/facebook_combined.txt'
            G = nx.read_edgelist(file, nodetype=int)
            G.name = 'Facebook'
        elif graph_number % number_of_graphs == 3:
            file = './code2/data/dimacs10-polblogs/edgelist.txt'
            G = nx.read_edgelist(file, nodetype=int)
            G.name = 'PolBlogs'
        elif graph_number % number_of_graphs == 4:
            G = nx.generators.community.LFR_benchmark_graph(1000, 3, 2, 0.1, average_degree=10, min_community=10, seed=42)
            G.name = 'LFR_1000'
        elif graph_number % number_of_graphs == 5:
            file = './code2/data/cora/cora.cites'
            G = nx.read_edgelist(file, nodetype=int, delimiter='\t')
            G.name = 'Cora'
        else:
            pass

        graph_number += 1

        # relabel nodes
        nodes = list(G.nodes())
        mapping = {}
        j = 0
        for i in nodes:
            mapping[i] = j
            j += 1
        G = nx.relabel_nodes(G, mapping, copy=True)

        main(model_save_path=_args.save,
             num_epochs=_args.epochs,
             learning_rate=_args.lr,
             G=G,
             load_weights_only=_args.load_weights_only,
             self_loops=_args.loops,
             algo=_args.algo,
             anomaly_param=_args.anomaly_param)

        print('#'*100)
        print(G.name + ' finished')
        print('#'*100)
