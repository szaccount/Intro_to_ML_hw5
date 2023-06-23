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
    Each hypothesis is represented by tuple (index, threshold, sign)
    where index is the index in the vector to compare to threshold and
    sign marks return sign for smaller than threshold.
    """
    hypotheses = []
    hypotheses_weights = []

    # Initializing distribution to be uniform
    dist = [1/len_train] * len_train

    for t in range(T):
        # returns chosen hypothesis and its error
        ht, epsilon_t = next_hypothesis(dist, X_train, max_per_coor, y_train)
        







##############################################
# You can add more methods here, if needed.





##############################################


def main():
    T = 80
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

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
                max_vals[i] = x[i]
    
    return max_vals

############### Move to the methods place under run_adaboost
###############!!!!!!!!!!!!!!!!!!!!!!!!!!!###############

if __name__ == '__main__':
    main()



