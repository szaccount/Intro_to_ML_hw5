#################################
# Your name: Sean Zaretzky id: 209164086
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    max_per_coor = get_max_count_per_coor(X_train)
    len_train = len(X_train)
    
    """
    Each hypothesis is represented by tuple (sign, index, threshold)
    where index is the index in the vector to compare to threshold and
    sign represents the return sign for smaller than threshold.
    """
    hypotheses = []
    hypotheses_weights = []

    # Initializing distribution to be uniform
    dist = [1/len_train] * len_train

    for t in range(T):
        # returns chosen hypothesis and its error
        ht, epsilon_t = next_hypothesis(dist, X_train, max_per_coor, y_train)
        print(f"Next weak hypothesis is: {ht} with epsilon: {epsilon_t}") # delete
        wt = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        print(f"{wt=}") # delete
        dist = calc_updated_weights(dist, wt, X_train, y_train, ht)
        ##### delete
        part_dist = dist[:10]
        print(f"{part_dist=}")
        ##### delete
        hypotheses.append(ht)
        hypotheses_weights.append(wt)

    print(f"{hypotheses=}")
    print("!!!!!!!!!!!!!!!!!!!!!")
    print(f"{hypotheses_weights=}")
    return hypotheses, hypotheses_weights






##############################################
# You can add more methods here, if needed.





##############################################


def main():
    T = 80
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    assert len(X_train) == len(y_train) # delete this line

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    # You can add more methods here, if needed.



    ##############################################


###############!!!!!!!!!!!!!!!!!!!!!!!!!!!###############

def get_max_count_per_coor(X_vec):
    """
    Returns max value appearing for every coordinate in the passed collection of X vectors.
    Used for upper bound on theta for each coordinate of the X_train vectors when training on them.
    """
    x_dim = len(X_vec[0])
    max_vals = [0] * x_dim
    for x in X_vec:
        for i in range(x_dim):
            if max_vals[i] < x[i]:
                max_vals[i] = int(x[i])
    
    return max_vals

def next_hypothesis(dist, X_vec, max_per_coor, y_vec):
    """
    Finds the best weak hypothesis to return based on the passed X_vec and distribution dist.
    
    Param max_per_coor: precalculation of the max value appearing in every coordinate of the X_vec vectors.
    Returns: best found weak hypothesis represented as tuple (sign, index, threshold) and its error.
    In the hypothesis representation, index is the index in the vector to compare to threshold and
    sign represents the return sign for smaller than threshold.
    """

    rv_hypo = (1, 0, 1)
    rv_error = 2 # should be smaller than 0.5 for best weak hypothesis

    sample_size = len(X_vec)
    for indx in range(sample_size):
        for thresh in range(1, max_per_coor[indx]): # maybe try to include the boundries (0, max_per_coor[indx] + 1)
            for sign in (1, -1):
                hypo_try = (sign, indx, thresh)
                error_for_hypo = calc_hypothesis_error(dist, X_vec, y_vec, hypo_try)
                if (error_for_hypo < rv_error):
                    rv_error = error_for_hypo
                    rv_hypo = hypo_try
    
    return rv_hypo, rv_error 

def calc_hypothesis_error(dist, X_vec, y_vec, hypothesis):
    """
    Calculates the weighted error for the passed hypothesis.
    Hypothesis represented as tuple (sign, index, threshold).
    """
    sample_size = len(X_vec)
    rv = 0
    for i in range(sample_size):
        zero_one_error = calc_zero_one_error(X_vec[i], y_vec[i], hypothesis)
        rv += (dist[i] * zero_one_error)
    
    return rv

def calc_zero_one_error(x, y, hypothesis):
    hypothesis_ans = eval_hypothesis(x, hypothesis)

    if hypothesis_ans == y:
        return 0
    else:
        return 1
    
def eval_hypothesis(x, hypothesis):
    """
    Returns value of hypothesis on x.
    """
    sign, indx, threshold = hypothesis
    if x[indx] <= threshold:
        hypothesis_ans = sign
    else:
        hypothesis_ans = -1 * sign
    
    return hypothesis_ans


def calc_updated_weights(dist, wt, X_vec, y_vec, ht):
    """
    Returns updated weights of ditribution based on passed results from previous round
    """
    rv = []
    sample_size = len(X_vec)
    denom = 0
    neg_wt = -1 * wt
    for j in range(sample_size):
        denom += dist[j] * np.exp(neg_wt * y_vec[j] * eval_hypothesis(X_vec[j], ht))
    
    for i in range(sample_size):
        nominator = dist[i] * np.exp(neg_wt * y_vec[i] * eval_hypothesis(X_vec[i], ht))
        rv.append(nominator / denom)
    
    return rv


############### Move to the methods place under run_adaboost
###############!!!!!!!!!!!!!!!!!!!!!!!!!!!###############

if __name__ == '__main__':
    main()



