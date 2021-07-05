
#Add egde
#check the prediction and accuracy function
#check cross validation
#check pruning

# version 1.3
# We will run your code with Python 3.6.9 on linux.student.cs.uwaterloo.ca 

import math
from anytree import AnyNode, Node, RenderTree
import global_var # global variables in dt_io.py
import numpy as np


def entropy(dist):
    """
    Takes a vector (dist) representing a probability distribution,
    and returns the entropy of the distribution, 
    calculated using the formula -Sigma_{p in dist}p*log2(p).

    Used in choose_feature_split.

    :param dist: a probability vector
    :type dist: List[float]
    :require dist: sum up to 1, elements in [0,1]
    :Example: [0.1,0,0.9]
    :return: the entropy of dist
    :rtype: float
    """
    e = 0.0
    for p in dist:
        if p !=0:
            if p !=1:
                e += p * math.log2(p)
        entropy = -e

    return entropy



def get_splits(examples, feature):
    """
    Given a set of examples and a feature, 
    determines a list of the potential split point values for the feature.
    (Given a sorted list of feature values, each split point value 
    is a mid-way value between two consecutive feature values.)
    
    For each split point value, return a list of integers, one for each class/label.
    The integer at index k in the list is the number of examples where 
    all the examples have the kth class/label value and 
    the feature value is less than the split point value.

    This function returns a dictionary 
    where each key is a possible split point value
    and each value is a list of integers describing the number of examples
    with the correspnding class/label where the feature value is less
    than the split point value.

    Used in choose_feature_split.
    
    :param examples: the 2d data array. 
    :type examples: List[List[Any]]

    :param feature: target feature
    :type feature: str
    :return: potential split point values and how many instances are less than
    the split point values for each class in list format
    :rtype: dict[float:List[int]]

    :Example: {2.65: [0, 1], 3.05: [1, 1], 3.15: [2, 2], 3.35: [3, 4]}
    There are 4 possible split point values.  
    There are 2 classes 0 and 1, thus the values are length 2 lists. 

    For example, among all the examples with feature <= 2.65, 
    there are 0 examples with label 0 and 1 example with label 1.

    Among all the examples with feature <= 3.35, 
    there are 3 examples with label 0 and 4 examples with label 1.
    """ 
    
    index = global_var.index_dict[feature + "_index"]
    count_label_initial = [0]*len(global_var.index_label_dict.keys())
    examples.sort(key=lambda x:x[-1])
    examples.sort(key=lambda x:x[index])
    result = {}
    for i in range(len(examples)-1):
        count_label = count_label_initial.copy()
        if examples[i][index] != examples[i+1][index]:
            if examples[i][-1] != examples[i+1][-1]:
                split = (examples[i][index]+examples[i+1][index])/2
                for j in range((i+1)):
                    if examples[j][index]<=split:
                        count_label[examples[j][-1]]+=1
                result[split] = count_label
    return result



def choose_feature_split(examples, features):
    """
    Given a set of examples (examples) and a set of features (features), 
    this function returns the feature with the split value 
    that has the largest expected information gain.

    Used in split_node.

    :param examples: the 2d data array.
    :type examples: List[List[Any]]
    :param features: remaining features
    :type features: List[str]
    :return: the chosen feature and the split value 
             with the max expected info gain
    :rtype: str, float
    """  
    all_features_number = [x[-1] for x in examples]
    a= np.bincount(all_features_number)
#    print("all_features_number",all_features_number)
    all_feature_count = np.pad(a,(0,len(global_var.index_label_dict.keys())-len(a)),'constant')
#    print("all_feature_count",all_feature_count)
    best_feature_entropy ={}
    best_feature_entropy_split = {}
    for feat in features:
        index = global_var.index_dict[feat + "_index"]
#        print("feat: ",feat)
        examples.sort(key=lambda x:x[-1])
        examples.sort(key=lambda x:x[index])
        splits = get_splits(examples,feat)

        sp =[]
        for key in splits:
            sp.append(key)

        if not sp:
            continue
        
        entropy_dict ={}
        for i in sp:
            val = np.array(splits[i])/sum(np.array(splits[i]))
            val = val.tolist()
#            print("val",val)
            remain_val = all_feature_count - splits[i]
            remain_val_ans = np.array(remain_val)/sum(np.array(remain_val))
            remain_val_ans = remain_val_ans.tolist()
#            print("remain_val",remain_val_ans)
            entropy1 = entropy(val)
#            print("entropy1 ", entropy1)
            entropy2 = entropy(remain_val_ans)
#            print("entropy2 ",entropy2)
            entropy_ans = ((sum(np.array(splits[i]))/len(examples))*entropy1)+((sum(np.array(remain_val))/len(examples))*entropy2)
            entropy_dict[i]=entropy_ans
       # print("entropy_dict: ",entropy_dict)

        best_split_entropy = min(entropy_dict, key=entropy_dict.get)
#        print("best_split_entropy",best_split_entropy)
        best_feature_entropy[feat] = entropy_dict[best_split_entropy]
        best_feature_entropy_split[feat] = best_split_entropy

#    print("best_feature_entropy",best_feature_entropy)
    if not best_feature_entropy:
        return None, -1
    best_entropy_feature = min(best_feature_entropy, key=best_feature_entropy.get) #Should be string
    best_entropy_feature_split = best_feature_entropy_split[best_entropy_feature] #Should be float
   # return best_entropy_feature, best_entropy_feature_split 
    
    return best_entropy_feature, best_entropy_feature_split




def split_examples(examples, feature, split):
    """
    Splits examples into two sets by a feature and a split value.
    Returns two sets of examples.
    The first set has examples where feature value <= split value
    The second set has examples where feature value is > split value.  

    Used in split_node.

    :param examples: the 2d data array.
    :type examples: List[List[Any]]
    :param feature: the feature name
    :type feature: str
    :param split: the split value
    :type split: float
    :return: two sets of examples 
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    less_than_split = []
    more_than_split = []
    index = global_var.index_dict[feature + "_index"]
    examples.sort(key=lambda x:x[index])
    if len(examples) == 0:
        return examples, None
    else:
        for example in examples:
            if example[index] <= split:
                less_than_split.append(example)
            elif example[index] > split:
                more_than_split.append(example)
        return less_than_split, more_than_split



def split_node(cur_node, examples, features, max_depth):
    """
    Splits cur_node and grows the tree until the max_depth.
    If cur_node level has reached max_depth, 
    this function makes a leaf node with majority decision 
    and returns the leaf node.
    This function calls itself recursively on 
    its left and right subtrees.

    Used in learn_dt.

    :param cur_node: the current node
    :type cur_node: Node
    :param examples: the 2d data array
    :type examples: List[List[Any]]
    :param features: features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    :return: the current node as the root of the tree
    :rtype: Node
    """ 
    depth = cur_node.dep
    labels = []
    feat = []
    for example in examples:
        labels.append(example[-1])
    for feature in features:
        feat.append(feature)
    #1. if all examples are in the same class then
    if all(x == labels[0] for x in labels):
        cur_node.fd = labels[0]
        cur_node.leaf = True
        return cur_node
    #2. else if no features left then
    elif len(feat) == 0:
        ans = None
        count = 0
        for c in labels:
            if count != 0:
                count += 1 if ans == c else -1
            else: # count == 0
                ans = c
                count = 1
        if labels.count(ans) > len(examples) // 2 :
            cur_node.fd = ans
            cur_node.leaf = True
            return cur_node
    # check the majority
    #   return ans if labels.count(ans) > len(examples) // 2 
    elif len(examples) == 0:
        par = cur_node.parent
        majority = par.majority
        cur_node.majority = majority
        return cur_node

    elif depth == max_depth:
        cur_node.leaf = True
        cur_node.fd = cur_node.majority
        return cur_node

     ####return majoriuty at parent node
    else:
        left_child_split, right_child_split = split_examples(examples, cur_node.fd, cur_node.split)

        all_features_number_left = [x[-1] for x in left_child_split]
        a= np.bincount(all_features_number_left)
        all_feature_count_left = np.pad(a,(0,len(global_var.index_label_dict.keys())-len(a)),'constant')

        all_feature_count_left = (all_feature_count_left/sum(np.array(all_feature_count_left))).tolist()
        entropy_left = entropy(all_feature_count_left)

        info_gain_left = cur_node.entropy - entropy_left

        all_features_number_right = [x[-1] for x in right_child_split]
        a= np.bincount(all_features_number_right)
        all_feature_count_right = np.pad(a,(0,len(global_var.index_label_dict.keys())-len(a)),'constant')

        all_feature_count_right = (all_feature_count_right/sum(np.array(all_feature_count_right))).tolist()
        entropy_right = entropy(all_feature_count_right)

        info_gain_right = cur_node.entropy - entropy_right

        
        if depth <= max_depth:

            if len(left_child_split) <=1:
                maj_left = find_majority(left_child_split)
                left_child = Node(name=cur_node.name+"left"+str(depth), parent=cur_node, fd=maj_left, leaf=True, dep=(depth+1), majority=maj_left, entropy=entropy_left, infoGain = info_gain_left, edge="<="+str(cur_node.split))
            elif len(left_child_split) >1:
                left_feature, left_split_value = choose_feature_split(left_child_split, features)
                if left_feature==None:
                    maj_left = find_majority(left_child_split)
                    left_child = Node(name=cur_node.name+"left"+str(depth), parent=cur_node, fd=maj_left, leaf=True, split=left_split_value, dep=(depth+1), majority=maj_left, entropy=entropy_left, infoGain = info_gain_left, edge="<="+str(cur_node.split))
                else:
                    maj_left = find_majority(left_child_split)
                    left_child = Node(name=cur_node.name+"left"+str(depth), parent=cur_node, fd=left_feature, leaf=False, split=left_split_value, dep=(depth+1), majority=maj_left,edge="<="+str(cur_node.split), entropy=entropy_left, infoGain=info_gain_left)
                    l = split_node(left_child, left_child_split, features, max_depth)


            if len(right_child_split) <=1:
                maj_right = find_majority(right_child_split)
                right_child = Node(name=cur_node.name+"right"+str(depth), parent=cur_node, fd=maj_right, leaf=True, dep=(depth+1), majority=maj_right, entropy=entropy_right, infoGain=info_gain_right, edge = ">"+str(cur_node.split))

            elif len(right_child_split) >1:
                right_feature, right_split_value = choose_feature_split(right_child_split, features)
                if right_feature==None:
                    maj_right = find_majority(right_child_split)
                    right_child = Node(name=cur_node.name+"right"+str(depth), parent=cur_node, fd=maj_right, leaf=True, split=right_split_value, dep=(depth+1), majority=maj_right, entropy=entropy_right, infoGain=info_gain_right, edge=">"+str(cur_node.split))
                else:
                    maj_right = find_majority(right_child_split)
                    right_child = Node(name=cur_node.name+"right"+str(depth), parent=cur_node, fd=right_feature, leaf=False, split=right_split_value, dep=(depth+1), majority=maj_right, edge=">"+str(cur_node.split), entropy=entropy_right, infoGain=info_gain_right)
                    r = split_node(right_child, right_child_split, features, max_depth)


        return cur_node



def learn_dt(examples, features, label_dim, max_depth=math.inf):
    """
    Given examples, features, number of label values and the max depth,
    learns a decision tree to classify the examples,
    and returns the root node of the tree.

    label_dim is the last return value of read_data in dt_io.py.
    
    This is the wrapper function for split_node.

    :param examples: the 2d data array.
    :type examples: List[List[Any]]
    
    :param features: features
    :type features: List[str]

    :param label_dim: the number of possible label values
        the last return value of read_data in dt_io.py.
    :type label_dim: int

    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf

    :return: the tree
    :rtype: Node
    """ 
    feature, split_value = choose_feature_split(examples, features)
    maj_dec = find_majority(examples)
    all_features_number = [x[-1] for x in examples]
    a= np.bincount(all_features_number)
    all_feature_count = np.pad(a,(0,len(global_var.index_label_dict.keys())-len(a)),'constant')


    all_feature_count = (all_feature_count/sum(np.array(all_feature_count))).tolist()
    entropy_root = entropy(all_feature_count)


    root = Node(name="root", parent=None, fd=feature, leaf=False, split=split_value, dep=1, majority=maj_dec, entropy=entropy_root, edge="")
    #print(RenderTree(root))
    split_node(root, examples, features, max_depth)
    return root
 



def predict(tree, example, max_depth=math.inf):
    """
    Given a decision tree, an example and a max depth,
    returns a label for the example 
    based on the decision tree up to the max depth.

    If we haven't reached a leaf node at the max depth, 
    we will return the majority decision at the last node.

    This function calls itself recursively.

    Used in get_prediction_accuracy.
    
    :param tree: a decision tree
    :type tree: Node
    :param example: one example
    :type example: List[Any]
    :Example: [5.1, 3.5, 1.4, 0.2, 0.0] is an example 
              with 5 feature values and a label value of 0.0.
    :param max_depth: the max depth
    :type param max_depth: int, default math.inf
    :return: the decision for the given example
    :rtype: int
    """ 

    if tree.dep == max_depth:
        return tree.majority
    else:
        if tree.leaf:
            return tree.fd
            
            #print(tree.fd)
        else:
            feature_id = tree.fd
            index = global_var.index_dict[feature_id + "_index"]

            if example[index] <= tree.split:
                return predict(tree.children[0], example,max_depth)
            elif example[index] > tree.split:
                return predict(tree.children[1],example,max_depth)




def get_prediction_accuracy(tree, examples, max_depth=math.inf):
    """
    Calculates the prediction accuracy for the given examples 
    based on the given decision tree up to the max_depth. 

    If we have not reached a leaf node at max_depth, 
    return the majority decision at the node.

    Used in cv.py.

    :param tree: a decision tree
    :type tree: Node
    :param examples: a 2d data array containing set of examples.
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type param max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the tree
    :rtype: float
    """ 
   # true_values = 0
   # accuracy = 0.0
   # for example in examples:
   #     prediction = predict(tree, example, max_depth)
   #     if index_label_dict[prediction] == example[-1]:
   #         true_values = true_values + 1
   # accuracy = true_values / len(examples)



    true_values = 0
    accuracy = 0.0
    for example in examples:
        prediction = predict(tree, example, max_depth)
        if prediction == example[-1]:
            true_values = true_values + 1
    accuracy = true_values / len(examples)
    return accuracy



def post_prune_node_ig(cur_node, post_info_gain):
    """
    Post prunes the tree with cur_node as the root of the tree
    using the expected info gain criterion.

    post_info_gain is the pre-defined value of the minimum info gain.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents. If the expected info gain at a leaf parent 
    is smaller than the pre-defined value, convert the leaf parent into a leaf node.
    Repeat until the expected info gain at every leaf parent is greater than
    or equal to the pre-defined value of the minimum info gain.

    :param cur_node: the current node
    :type cur_node: Node
    :param post_info_gain: the minimum info gain
    :type post_info_gain: float
    :return: the post-pruned tree with cur_node as the root
    :rtype: Node
    """ 
    if cur_node.leaf:
        return cur_node
    elif cur_node.children[0].leaf == True and cur_node.children[1].leaf == True:
        #This is a leaf parent
        if cur_node.infoGain < post_info_gain:
            cur_node.children[0].parent = None
               # node.children[1].parent = None
            cur_node.leaf =True
            cur_node.fd = cur_node.majority
            #print(cur_node)
            return post_prune_node_ig(cur_node.children[0], post_info_gain)
    else:
        return post_prune_node_ig(cur_node.children[0], post_info_gain)
        
                #return post_prune_node_ig(cur_node, post_info_gain)
    




def find_majority(ed):
    maj = [row[-1] for row in ed]
    return max(set(maj), key=maj.count)






