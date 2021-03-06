"""
this is where the magic happens
this file will decode the message encoded by the substitution cipher
"""
import argparse
import random
import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from data_mining import get_one_point_statistics, get_transition_matrix, \
    load_data, legal_chars, chars_to_index, index_to_chars


def load_encoded_message(file):
    """
    load the encoded message from a file into a long string
    """
    with open(file, 'r') as f:
        data = f.read()
    return data


def swap_element(permutation):
    """
    swap the elements at index i and j in permutation
    this swapped permutation is defined as a neighbor of the current
    permutation in the MCMC random graph
    """
    # choose two indices to swap
    i, j = np.random.choice(len(permutation), 2, replace=False)

    neighbor = list(deepcopy(permutation))
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return ''.join(neighbor)


def energy_func(encoded_msg, one_point_stat, transition_mat, permutation):
    """
    get the energy of a certain permutation
    the energy is defined as the negative log likelihood
    likelihood is P(permutation | encoded_msg)
    """
    permutation_inverse = lambda b: legal_chars[permutation.index(b)]
    likelihood  = 0
    # liklihood of the first letter
    likelihood += math.log(one_point_stat[permutation_inverse(encoded_msg[0])])
    # likelihood of the rest of the transition pairs
    for i in range(1, len(encoded_msg)-1):
        x, y = permutation_inverse(encoded_msg[i]), permutation_inverse(encoded_msg[i+1])
        x_idx, y_idx = chars_to_index[x], chars_to_index[y]
        try:
            likelihood += math.log(transition_mat[x_idx, y_idx])
        except ValueError:
            # transition probability is 0
            likelihood += math.log(1e-16)
    return -likelihood


def mcmc(num_iters, beta, encoded_msg, one_point_stat, transition_mat,
         plot_save_path, start_state='abcdefghijklmnopqrstuvwxyz ', plot_every=1):
    """
    markov chain monte carlo
    try to sample the correct permutation according to a Gibbs distribution
    make a plot of the energy of walk throught the random graph, record the energy
    every `plot_every` iterations
    """
    current_state = start_state
    energy = lambda state: energy_func(encoded_msg, one_point_stat, transition_mat, state)
    energy_list = []
    # start iteration
    for i in range(num_iters):
        print(f"iteration {i}. energy: {energy(current_state)}. currnent: {current_state}")

        # record the energy
        if i % plot_every == 0:
            energy_list.append(energy(current_state))

        # choose a neighbor of current state uniformly at random
        next_state = swap_element(current_state)

        # accept or reject according to energy change
        energy_diff = energy(next_state) - energy(current_state)
        if energy_diff < 0:
            current_state = next_state
        else:
            accept_prob = math.exp(-beta * energy_diff)
            if random.random() < accept_prob:
                # accept
                current_state = next_state
            else:
                # reject
                pass
        
    # plot the energy trajectory
    plt.plot(plot_every * np.arange(len(energy_list)), energy_list)
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.title('Energy of Walk Through Metropolis Graph')
    plt.savefig(plot_save_path)
    
    return current_state


def decode_message(encoded_msg, permutation, save_to=None):
    """
    decode a long string
    args:
        encoded_msg: a long string encoded by substitution cipher
        permutation: a 27-char string of all legal chars
    return:
        decoded_msg: a long string decoded by substitution cipher
    """
    permutation_inverse = lambda b: legal_chars[permutation.index(b)]
    decoded_msg = ''.join(map(permutation_inverse, encoded_msg))

    # save the result
    if save_to:
        with open(save_to, 'w') as f:
            f.write(decoded_msg)

    return decoded_msg


def main():
    """
    put everything together
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='pride-and-prejudice.txt', 
                            help='the file containing the data of true English language')
    parser.add_argument('--code', type=str, default='f_45.txt',
                            help='the file containing the encoded message')
    parser.add_argument('--save_path', type=str, default='decoded.txt',
                            help='the file to save the decoded message')
    parser.add_argument('--beta', type=float, default=1.0,
                            help='the beta parameter of the Gibbs distribution')
    parser.add_argument('--num_iters', type=int, default=100000,
                            help='the number of iterations to run MCMC')
    parser.add_argument('--seed', type=int, default=0,
                            help='the seed for random number generator')
    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load the code
    encoded = load_encoded_message(args.code)

    # data mining on the true English language
    true_language_data = load_data(args.data)
    one_point_stat = get_one_point_statistics(true_language_data)
    transition_matrix = get_transition_matrix(true_language_data)

    # run MCMC
    final_permutation = mcmc(
        num_iters=args.num_iters,
        beta=args.beta,
        encoded_msg=encoded,
        one_point_stat=one_point_stat,
        transition_mat=transition_matrix,
        plot_save_path=args.save_path[:-4] + '_energy.png',
    )
    print(final_permutation)

    # decode the message
    real_msg = decode_message(encoded, final_permutation, save_to=args.save_path)
    print(real_msg)


if __name__ == '__main__':
    main()
