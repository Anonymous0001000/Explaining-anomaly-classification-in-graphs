import numpy as np
from sklearn.metrics import confusion_matrix


def get_precision_and_recall(graph_name, anomaly_param, importance_type, algo):
    precision_array = []
    recall_array = []
    if algo == 'our':
        # file to read
        result_file = "./results/result_" + str(graph_name) + "_A_" + str(anomaly_param) + ".txt"
    else:
        result_file = 'NULL'
    precision = 0
    recall = 0
    number_of_nodes = 0
    with open(result_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            if i % 4 == 0:
                # first line is the ground truth
                ground_truth1 = np.array([line[j] for j in range(1, len(line))], dtype=float)
            elif i % 4 == 1:
                # second line is the second ground-truth which is different in case of disjunction
                ground_truth2 = np.array([line[j] for j in range(1, len(line))], dtype=float)
            elif i % 4 == 2:
                # third line is the prediction
                feature_pred1 = np.array([line[j] for j in range(1, len(line))], dtype=float)
            elif i % 4 == 3:
                # fourth line is blank, we compute precision and recall here
                number_of_nodes += 1
                F = len(ground_truth1)
                # ground-truth1, ground_truth2 and feature_pred1 have same size
                predicted_features = np.array([0 for i in range(F)])
                predicted_features2 = np.array([0 for i in range(F)])
                # there are two predicted_features, one for each ground-truth
                if importance_type == 'all':
                    # if we want all features whose importance is more than zero
                    importance = np.nonzero(feature_pred1)[0]
                    importance2 = np.nonzero(feature_pred1)[0]
                elif importance_type == 'best':
                    # if we want the most important features
                    number_important_features = min(int(np.sum(ground_truth1)), len(np.nonzero(feature_pred1)[0]))
                    threshold = np.sort(feature_pred1)[-number_important_features]
                    importance = np.array(feature_pred1 >= threshold, dtype=int)
                    importance = np.nonzero(importance)[0]
                    number_important_features2 = min(int(np.sum(ground_truth2)), len(np.nonzero(feature_pred1)[0]))
                    threshold2 = np.sort(feature_pred1)[-number_important_features2]
                    importance2 = np.array(feature_pred1 >= threshold2, dtype=int)
                    importance2 = np.nonzero(importance2)[0]
                predicted_features[importance] = 1
                predicted_features2[importance2] = 1

                # compute confusion matrices
                conf1 = confusion_matrix(predicted_features, ground_truth1)
                conf2 = confusion_matrix(predicted_features2, ground_truth2)
                valid_ground_truth1 = 0
                valid_ground_truth2 = 0
                # no disjunction for A0, A3, A4 and A5 so both ground_truth are valid (they are the same)
                if anomaly_param == 0 or anomaly_param == 3 or anomaly_param == 4 or anomaly_param == 5:
                    valid_ground_truth1 = 1
                    valid_ground_truth2 = 1
                elif anomaly_param == 1:
                    for k in range(len(ground_truth1)-5):
                        if k%5==1 and ground_truth1[k] == 1:
                            valid_ground_truth1 = 1
                    for k in range(len(ground_truth2)-5):
                        if k%5==2 and ground_truth2[k] == 1:
                            valid_ground_truth2 = 1
                # in case of disjunction, check which condition is the right one
                elif anomaly_param == 2:
                    bleu = 0
                    vert = 0
                    rouge = 0
                    noir = 0
                    for k in range(len(ground_truth1)-5):
                        if k%5==0 and ground_truth1[k] == 1:
                            bleu = 1
                        if k%5 == 1 and ground_truth1[k] == 1:
                            vert = 1
                        if k%5== 2 and ground_truth2[k] == 1:
                            rouge = 1
                        if k%5==4 and ground_truth2[k] == 1:
                            noir = 1
                    if bleu and vert:
                        valid_ground_truth1 = 1
                    if rouge and noir:
                        valid_ground_truth2 = 1
                elif anomaly_param == 6:
                    bleu = 0
                    noir = 0
                    if ground_truth1[-1] == 1:
                        valid_ground_truth1 = 1
                    for k in range(len(ground_truth2)-5):
                        if k%5 == 0 and ground_truth2[k] == 1:
                            bleu = 1
                        if k%5 == 4 and ground_truth2[k] == 1:
                            noir = 1
                    if bleu and noir:
                        valid_ground_truth2 = 1

                # if confusion matrix is (1, 1), make it (2, 2) by adding zeros
                if conf1.shape == (1, 1):
                    conf1 = np.array([[conf1[0][0], 0], [0, 0]])
                if conf2.shape == (1, 1):
                    conf2 = np.array([[conf2[0][0], 0], [0, 0]])

                # compute precision and recall
                precision1 = conf1[1][1] / (conf1[1][1] + conf1[1][0])
                recall1 = conf1[1][1] / (conf1[1][1] + conf1[0][1])
                precision2 = conf2[1][1] / (conf2[1][1] + conf2[1][0])
                recall2 = conf2[1][1] / (conf2[1][1] + conf2[0][1])
                if conf1[1][1] == 0:
                    precision1 = 0
                    recall1 = 0
                if conf2[1][1] == 0:
                    precision2 = 0
                    recall2 = 0
                try:
                    fmeasure1 = precision1 * recall1 / (precision1 + recall1)
                except:
                    fmeasure1 = 0
                try:
                    fmeasure2 = precision2 * recall2 / (precision2 + recall2)
                except:
                    fmeasure2 = 0

                # if both ground_truth are valid, get the best explanation with fmeasure
                # results are stored in precision_array and recall_array
                if valid_ground_truth1 and valid_ground_truth2:
                    if fmeasure1>=fmeasure2:
                        precision_array.append(precision1)
                        recall_array.append(recall1)
                    else:
                        precision_array.append(precision2)
                        recall_array.append(recall2)

                elif valid_ground_truth1:
                    precision_array.append(precision1)
                    recall_array.append(recall1)

                elif valid_ground_truth2:
                    precision_array.append(precision2)
                    recall_array.append(recall2)

                else:
                    print('Error')
                    stop

    precision_array = np.array(precision_array)
    recall_array = np.array(recall_array)
    # return mean and std deviation of precision and recall
    return np.mean(precision_array), np.std(precision_array), np.mean(recall_array), np.std(recall_array)


if __name__ == '__main__':
    # names of the graphs
    names = ['Erdos-Renyi_10000', 'dancer_10000', 'Facebook', 'PolBlogs', 'LFR_1000', 'Cora']
    # algorithms involved
    algos = ['our']
    for anomaly_param in [0, 1, 2, 3, 4, 5, 6]:
        for graph_name in names:
            print(anomaly_param, graph_name)
            # initialize with -1 to check errors
            K = len(algos)
            precision_means = [-1 for i in range(K)]
            precision_stds = [-1 for i in range(K)]
            recall_means = [-1 for i in range(K)]
            recall_stds = [-1 for i in range(K)]
            for algo in algos:
                try:
                    precision_mean, precision_std, recall_mean, recall_std = get_precision_and_recall(graph_name, anomaly_param, importance_type='all', algo=algo)
                except:
                    precision_mean, precision_std, recall_mean, recall_std = -1, -1, -1, -1
                k = algos.index(algo)
                precision_means[k] = round(precision_mean, 2)
                precision_stds[k] = round(precision_std, 2)
                recall_means[k] = round(recall_mean, 2)
                recall_stds[k] = round(recall_std, 2)
                print(precision_mean, precision_std, recall_mean, recall_std)




















