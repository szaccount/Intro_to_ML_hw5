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
    hypotheses = [0] * T
    hypotheses_weights = [0] * T

    # Initializing distribution to be uniform
    dist = [1/len_train] * len_train

    for t in range(T):
        # returns chosen hypothesis and its error
        ht, epsilon_t = next_hypothesis(dist, X_train, max_per_coor, y_train)
        # print(f"weak hypothesis of round {t=} is: {ht} with epsilon: {epsilon_t}")
        wt = 0.5 * np.log((1 - epsilon_t) / epsilon_t)
        dist = calc_updated_weights(dist, wt, X_train, y_train, ht)
        hypotheses[t] = ht
        hypotheses_weights[t] = wt

    return hypotheses, hypotheses_weights



##############################################
# You can add more methods here, if needed.

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
        for thresh in range(0, max_per_coor[indx] + 1):
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
    sample_size = len(X_vec)
    nominators = np.zeros(sample_size)
    neg_wt = -1 * wt
    
    for i in range(sample_size):
        nominator = dist[i] * np.exp(neg_wt * y_vec[i] * eval_hypothesis(X_vec[i], ht))
        nominators[i] = nominator
    
    denom = nominators.sum()
    return np.divide(nominators, denom)

def calc_error(X_vec, y_vec, hypotheses, alpha_vals):
    sample_size = len(X_vec)
    num_iters = len(hypotheses)
    # Sum for each data point of values of hypotheses, updating per iteration
    sum_per_point = [0] * sample_size
    error_per_iter = []
    
    for t in range(num_iters):
        mispredictions = 0
        for i in range(sample_size):
            sum_per_point[i] += alpha_vals[t] * eval_hypothesis(X_vec[i], hypotheses[t])
            prediction = 1
            if (sum_per_point[i] < 0):
                prediction = -1
            if (prediction != y_vec[i]):
                mispredictions += 1
        
        error_per_iter.append(mispredictions / sample_size)
    
    return error_per_iter

def calc_exp_error(X_vec, y_vec, hypotheses, alpha_vals):
    sample_size = len(X_vec)
    num_iters = len(hypotheses)
    # Sum for each data point of values of hypotheses, updating per iteration
    sum_per_point = [0] * sample_size
    loss_per_iter = []
    
    for t in range(num_iters):
        loss_in_t = 0
        for i in range(sample_size):
            sum_per_point[i] += alpha_vals[t] * eval_hypothesis(X_vec[i], hypotheses[t])
            loss_in_t += np.exp(-y_vec[i] * sum_per_point[i])

        loss_per_iter.append(loss_in_t / sample_size)
    
    return loss_per_iter



##############################################


def main():
    T = 80
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    
    # For part a
    # train_error = calc_error(X_train, y_train, hypotheses, alpha_vals)
    # test_error = calc_error(X_test, y_test, hypotheses, alpha_vals)
    # iters_scale = list(range(T))
    # plt.figure(1)
    # plt.title("Training and test errors as a function of t (iteration)")
    # plt.xlabel("t")
    # plt.ylabel("Error")
    # plt.plot(iters_scale, train_error, '-', label="training error")
    # plt.plot(iters_scale, test_error, '-', label="test error")
    # plt.legend()
    # plt.show()

    # For part b
    # T = 10
    # print(f"Try {vocab[2]}") # delete
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # for i in range(T):
    #     print(f"Hypothesis {i}: {hypotheses[i]}, word at {hypotheses[i][1]}: {vocab[hypotheses[i][1]]}")

    # For part c
    # T = 80
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # train_error = calc_exp_error(X_train, y_train, hypotheses, alpha_vals)
    # test_error = calc_exp_error(X_test, y_test, hypotheses, alpha_vals)
    # iters_scale = list(range(T))
    # plt.figure(1)
    # plt.title("Training and test exponential loss as a function of t (iteration)")
    # plt.xlabel("t")
    # plt.ylabel("Error")
    # plt.plot(iters_scale, train_error, '-', label="training error")
    # plt.plot(iters_scale, test_error, '-', label="test error")
    # plt.legend()
    # plt.show()


    ##############################################


if __name__ == '__main__':
    main()



