
# coding: utf-8

# # The University of Melbourne, School of Computing and Information Systems
# # COMP30027 Machine Learning, 2019 Semester 1
# -----
# ## Project 1: Gaining Information about Naive Bayes
# -----
# ###### Student Name(s): Jonathan Reich
# ###### Python version: 3.5.6,
# ###### Submission deadline: 5pm, Mon 8 Apr 2019

#### Script to run N.B classifier. Just specify path to directory containing the csv files on the cmd line
#### and it will print out metrics onto the console.

from collections import defaultdict
import copy
import operator
import random
import math
import numpy as np
import os
import sys


# This function should open a data file in csv, and transform it into a usable format
def preprocess(file, split=0.0):
    """Opens a csv file of data and optionally transforms the data into 'training' and 'test' splits for
    'holdout' strategy. Also generates a dictionary structure that
    summarises the statistics about the number of instances of individual classes, as well as the number of instances
    found in the complete dataset, and the number of attributes found in the dataset.

    Args:
        file (csv file): dataset
        split (number): optional number that splits the data into partitions; default is set to 0 for training on all data.

    Returns:
        class_dict (dic): the number of instances found in the complete dataset.
        cleaned_data_train (list): two-dimensional list of lists containing the training data.
        cleaned_data_train (list): two-dimensional list of lists containing the test data.
        n_instances_test (number) : the number of instances found in the test dataset.
        class_dict (dic): dictionary giving counts of individual classes for the training dataset.
        test_dict (dic): dictionary giving counts of individual classes for the test dataset
        n_instances_tr (number): integer of the number of instances found in the training dataset.
        n_attrs (number): the number of attributes/properties found in a given instance.
    """

    n_attrs = 0
    f = open(file, 'r')
    # remember start of dataset for counting purposes.
    x = f.tell()
    first_line = f.readline().split(',')
    # read off number of attributes.
    for line in first_line[:-1]:
        n_attrs+=1
    # seek back to first line of dataset for counting purposes.
    f.seek(x)
    g = list(f)
    f.close()
    # count number of instances in whole dataset.
    n_instances = len(g)
    # if we want to use the 'holdout' strategy:
    if split > 0:
        cleaned_data_train = []
        cleaned_data_test = []
        test_dict=defaultdict(int)
        train_dict=defaultdict(int)
        # randomise the data before partitioning.
        random.shuffle(g)
        n_instances_tr = int(n_instances*split)
        # training instances split.
        training_data = g[:n_instances_tr]
        # test instance split.
        test_data = g[n_instances_tr:]
        n_instances_test = len(test_data)
        for line in training_data:
            # count classes in training set.
            train_dict[line.strip().split(",")[-1]] += 1
            # list of cleaned training data.
            cleaned_data_train.append(line.strip().split(','))
        for line in test_data:
            # count classes in test set.
            cleaned_data_test.append(line.strip().split(','))
            # list of cleaning test data.
            test_dict[line.strip().split(",")[-1]] += 1
        return cleaned_data_train, cleaned_data_test, n_instances_test, train_dict, test_dict, n_instances_tr, n_attrs

    cleaned_data = []

    class_dict=defaultdict(int)

    # No partitiong of data, so we will train on the full dataset:
    for line in g:
        # count classes in full dataset.
        class_dict[line.strip().split(",")[-1]] += 1
        # append the cleaned data to a list for further processing.
        cleaned_data.append(line.strip().split(','))

    return class_dict, n_instances, cleaned_data, n_attrs



# This function should build a supervised NB model
def train(training_data, n_instances, class_dict, n_attrs, ig_dic=False):
    """Trains the data from a preprocessed list and returns dictionaries of probabilities used to predict instances.

    Args:
        training_data (list): preprocessed dataset.
        n_instances (number): number of instances found in dataset.
        class_dict (dic): counts of classes.
        n_attrs (number): number of attributes found in a given instance.
        ig_dic (dictionary): gives a summary of counts of individual attributes (for use for information gain metric).

    Returns:
        prior (dic): dictionary of prior probabilities of classes.
        post (dic): dictionary of posterior probabilities of classes.
        training_data (list): training data in preprocessed form for next stage of the pipeline.
    """


    prior = defaultdict(int)
    # store the prior probabilities a dictionary.
    for key,val in class_dict.items():
        prior[key] = val/n_instances

    post=defaultdict(int)

    # iterate through the number of attributes.
    for attr in range(n_attrs):
        # create a nested dictionary for each attribute.
        temp_dict = defaultdict(int)
        for line in training_data:
            cls = line[-1]
            # create a key for each class in the nested dictionary.
            if cls not in temp_dict:
                temp_dict[cls] = defaultdict(int)

            # if the attribute itself doesn't already exist in the dictionary, count it as one.
            if line[attr] not in temp_dict[cls]:
                temp_dict[cls][line[attr]] = 1
            # otherwise, increment the count (this class, attribute pair already exists.)
            else:
                temp_dict[cls][line[attr]] += 1

        # set the posteior attribute to the counter from the 'temp_dict' object.
        post[attr] = temp_dict


        # iterate through each class in the dataset.
        for cls in prior.keys():
            # find aggregate of the values for each class.
            sum_value = sum(temp_dict[cls].values())
            for key, val in temp_dict[cls].items():
                # if parameter 'ig_dic' is true, then we will use the 'post' dic for
                # raw counts (to be used for the information gain function)
                if ig_dic:
                    temp_dict[cls][key] = val
                # Divide the class and attribute pair by the sum of the overall class.
                else:
                    temp_dict[cls][key] = val/sum_value

    return prior, post, training_data



# This function should predict the class for an instance or a set of instances, based on a trained model
def predict(prior, post, data):
    """Given instances of classes, predicts the class of each instance using the Naive Bayes' formula.

    Args:
        prior (dic): dictionary of prior probabilities of classes.
        post (dic): dictionary of posterior probabilities of classes.
        data (list): data on which to make predictions w/r/t the classes.

    Returns:
        final (dic): dictionary containing the classes of each predicted instance along with their associated probabilities.
        data (list): passes back data for use in evaluation metrics.
    """
    # value to be used to ensure probabilities don't hit 'zero', if a given attribute doesn't exist.
    EPSILON = 0.000001
    final = defaultdict(int)
    n_instances = len(data)
    # iterate over the data and create nested dictionaries to store predictions.
    for i, ins in enumerate(data):
        pred = defaultdict(int)
        for key, val in prior.items():
            # the prediction for the data point needs to be multiplied by the prior and posterior probabilities,
            # according to Bayes' formula.
            pred[key] = val
            for j in range(len(ins)-1):
                att = ins[j]
                # we will treat '?' as missing data.
                if att != '?':
                    # if the attribute isn't 'zero' or a '?' then multiply our prediction probability by the next entry.
                    if att in post[j][key]:
                        pred[key] *= post[j][key][att]
                    # otherwise, multiply our prediction probability by an infinitesimally small value.
                    else:
                        pred[key] *= EPSILON/n_instances
        # the predicted class for the instance will be the class that has the largest probability after looping through
        # the inner loops.
        final[i] = max(pred.items(), key=operator.itemgetter(1))

    return final, data


# This function should evaluate a set of predictions, in a supervised context
def evaluate(file, holdout_split=0):
    """Given a csv file, runs through the pipeline of preprocessing, training, predicting, then evaluating the dataset to
    obtain various metrics such as the overall accuracy (disregarding individual class accuracy), and the precision, recall
    and F1-score for each class. Also returns a 'ZeroR' Baseline.

    Args:
        file (csv file): dataset
        holdout_split (number): optional kwarg that splits the data into training and test partitions.

    Returns:
        accuracy (number): a numeric value that represents the ratio of the number of instances that matched (true positives)
        from the original/test dataset over the number of instances found in the set.
        baseline (number): 'zeroR' metric used as a baseline that classes each instant according to the majority class found in
        the dataset.
        confusion (dictionary of dictionaries): Summarises for each individual class in the dataset the precision, recall and
        f1-score (the harmonic mean of precision and recall).
    """

    if holdout_split:

        # sanity to check to ensure the holdout split is a valid value.
        assert ((holdout_split >= 0) and (holdout_split <= 1))

        # get preprocessed data.
        cleaned_data_train, cleaned_data_test, n_instances, dic, test_dict, first_split, n_attrs = preprocess(file, split=holdout_split)
        # get prior and post probability dictionary objects.
        prior,post,_ = train(cleaned_data_train, first_split, dic, n_attrs)
        # get the predicted classes, and get back the raw data in a list form for further evaluation.
        final, raw = predict(prior, post, cleaned_data_test)

    else:
        # same steps as above but for 'holdout' partitioning.
        dic,n_instances,cleaned_data,n_attrs = preprocess(file, split=holdout_split)
        prior,post,_ = train(cleaned_data, n_instances, dic, n_attrs)
        final, raw = predict(prior, post, cleaned_data)
    # get a baseline metric based on the majority class for the dataset (zero rule method)
    baseline = max(dic.values())/n_instances
    hits=0
    accuracy=0

    recall = defaultdict(int)
    prec = defaultdict(int)
    # if we have a hit then this is a true positive.
    for cls, instance in zip(final.values(), raw):
        if cls[0]==instance[-1]:
            hits+=1
            recall[cls[0]]+=1
        else:
        # otherwise, we have a FN or FP.
        # n.b we need to distinguish between FNs and FPs.
            prec[cls[0]]+=1
    # determines overall accuracy of the classifier
    accuracy = hits/n_instances
    final_metrics = defaultdict(int)
    # utility function to sort the values in lexicographical order.
    sorter_func = lambda x: x[0]
    # iterate through our stored classes to obtain metrics for each class in the dataset.
    for (a,b), (_, d), (_,f) in zip(sorted(recall.items(), key = sorter_func), sorted(prec.items(), key = sorter_func),
                                sorted(dic.items(), key = sorter_func)):
        # 'p' holds the precision metric (TP/TP+FP), 'r' holds the recall metric (TP/TP+FN)
        p,r = b/(b+d),  b/(b+f)
        # 'fs' is the f1-score, the harmonic mean of 'p' and 'r'.
        fs = 2*((p*r)/(p+r))
        # store the values in order in a dictionary of tuples.
        final_metrics[a] = (p, r, fs)

    return accuracy, baseline, final_metrics



# This function should calculate the Information Gain of an attribute or a set of attribute, with respect to the class
def info_gain(file):
    """Determines the information gain attribute for each attribute found in the csv file argument and returns
    the attribute found with the highest information gain.

    Args:
        file (csv file): dataset

    Returns:
        ig (dic): a dictionary containing each attribute in the dataset and their respective information gain,
        sorted in descending order.
        best_attr(key: value): the attribute that had the maximum information gain the dataset.
    """
    # preprocess the file to get the number of attributes and instances found in the dataset.
    dic,n_instances,cleaned_data,n_attrs = preprocess(file)
    # obtain a dictionary (ig_dic) of conditional counts of each attribute found in the dataset i.e. given each
    # class, how many instances of each attribute's values are found for a given class.
    prior,ig_dic,_= train(cleaned_data, n_instances, dic, n_attrs, ig_dic=True)
    # determine the class entropy (H(R)) using the raw counts from the prior probs.
    class_entropy = -(np.sum([g*np.log2(g) for g in list(prior.values())]))
    # Determine the raw counts of each attribute, regardless of their class.
    attr_dic= defaultdict(int)
    for i in range(n_attrs):
        attr_dic[i] = defaultdict(int)
        for line in cleaned_data:
            attr_dic[i][line[i]]+=1
    ar_orig = copy.deepcopy(attr_dic)
    # determine the probability of each attribute's value occurring in the dataset.
    probs_mean_info = attr_dic.copy()
    for key, ar in attr_dic.items():
        probs_mean_info[key] = ar
        for key2, val in ar.items():
            probs_mean_info[key][key2] = val/n_instances


    def log_probs(attr_dic, ig_dic, probs_mean_info, attr):
        """Determines the information gain for each attribute found in the dataset.

        Args:
            attr_dic (dic): dictionary containing the raw counts for each attribute found in the dataset.
            ig_dic (dic): dictionarying containing the conditional counts for each attribute's values found in the dataset.
            probs_mean_info (dic): dictionary containing the (prior) probabilities of finding a given attribute in
            the dataset.
            attr (number): the attribute which will be evaluated to determine the mean information.
            This is an integer corresponding to a key found in the attribute dictionary.

        Returns:
            mean_info (numeric value): the 'mean information' for the attribute.
        """
        # sanity check to ensure the user has inputted a valid key.
        assert attr in attr_dic.keys()
        # dictionary to contain the entropies for the attribute's values.
        logs_dic = defaultdict(int)
        mean_info = 0
        # iterate through the attribute dictionary, only looking at matching keys to our input attribute.
        for a, b in attr_dic[attr].items():
            for d in ig_dic[attr].values():
                for g in d.items():
                    # if the keys equal each other, then we multiply the prior prob of the attribute's value
                    # by the log of the prior prob of the attribute's value: sum from 1 to the number of types of attr vals
                    #where (attr='value') = -P(attr=val|class)*log(P(attr=val|class))
                    if a == g[0]:
                        logs_dic[a] += -(g[1]/b)*np.log2(g[1]/b)
        # the mean information is obtained by multiplying the entropies of the attribute's values by their respective
        # prior probabilties of being found in the dataset.
        for (a,b),(c,d) in zip(sorted(logs_dic.items()), sorted(probs_mean_info[attr].items())):
            mean_info+=b*d
        return mean_info

    # iterate through each attribute to find the 'information gain' for each attribute. We subtract the 'mean information'
    # for each attribute from H(R).
    ig = defaultdict(int)
    for i in range(n_attrs):
        ig_attr = class_entropy - log_probs(ar_orig, ig_dic, probs_mean_info, i)
        ig['attribute: ' + str(i)] = ig_attr
    # obtain the attribute which has the greatest information gain.
    best_attr = max(ig.items(), key=operator.itemgetter(1))

    # sort the attributes according to the information gain in descending order and return the attribute with the greatest
    # information gain.
    return sorted(ig.items(), reverse=True, key = lambda x: x[1]), best_attr


def main(dir, holdout_split=0):
# The 'dir" argument points to directory containing csv datasets. this function runs each individual function found in the
# pipeline and prints out evaluation metrics to the screen.

    count=0
    total=0.0
    avg = 0.0
    total_h=0.0
    avg_h=0.0
    for file in os.listdir(dir):
        if str(file).endswith('.csv'):
            count+=1
            acc, base, c = evaluate(file)
            _, best_attr = info_gain(file)
            total += acc
            avg = total / count
            if holdout_split:
                acc_h, _, cd = evaluate(file, holdout_split=holdout_split)
                total_h += acc_h
                avg_h=total_h/count
                print("TEST ACCURACY for", str(file).upper(), "is", acc, " \n and for BASELINE is : ", base, " \n and HOLDOUT is: ", acc_h)
                print("RELATIVE CLASSIFIER performance compared to BASELINE is:", 100 * (acc - base), "% DIFFERENCE")
                print("HOLDOUT performance compared to BASELINE is:", 100 * (acc_h - base), "% DIFFERENCE")
                print("ATTRIBUTE with GREATEST INFORMATION GAIN is: \n", best_attr)
                print("PRECISION, RECALL and F1-score for each class are: \n ", c.items())
                print("PRECISION, RECALL and F1-score for HOLDOUT for each class are: \n", cd.items())
                print()
            else:
                print("TEST ACCURACY for", str(file).upper(), "is", acc, " \n and for BASELINE is : ", base, "\n")
                print("RELATIVE CLASSIFIER performance compared to BASELINE is:", 100 * (acc - base), "% DIFFERENCE")
                print("ATTRIBUTE with GREATEST INFORMATION GAIN is: \n", best_attr)
                print("PRECISION, RECALL and F1-score for each class are: \n ", c.items())
                print()
    if avg_h>0:
        print("MEAN ACCURACY is", avg, "and for HOLDOUT it is:", avg_h)
        sys.exit(0)
    else:
        print("MEAN ACCURACY is", avg)
        sys.exit(0)


# Will run evaluation metrics printed to screen.
# Optionally, the user can specify their holdout split as a second argument on the command line.
if __name__ == "__main__":
    if len(sys.argv)==3:
        main(sys.argv[1], float(sys.argv[2]))
    else:
        main(sys.argv[1])
