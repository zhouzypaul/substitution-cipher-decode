"""
do data mining and obtain the one-point and two-point statistics
about the English language, and then calculate the likelihood function
"""
import re
import numpy as np
import matplotlib.pyplot as plt


legal_chars = 'abcdefghijklmnopqrstuvwxyz '
chars_to_index = {c:i for i, c in enumerate(legal_chars)}
index_to_chars = {i:c for i, c in enumerate(legal_chars)}


def load_data(file='pride-and-prejudice.txt'):
    """
    load the data from the file
    remove all the punctuations and make all te words lower case
    only leave characters a-z and space

    return: one long string
    """
    with open(file, 'r') as f:
        data = f.read()
    # to lowercase
    data = data.lower()
    # filter out white spaces
    data = data.replace('\n', ' ')
    data = data.replace('\r', ' ')
    data = data.replace('\t', ' ')
    # only keep the legal characters
    data = ''.join([c for c in data if c in legal_chars])
    # merge white spaces
    data = re.sub(' +', ' ', data)

    return data


def get_one_point_statistics(data):
    """
    get the one-point statistics about the English language from input data
    args:
        data: a long string
    return:
        a dictionary with keys as characters and values as the number of times
    """
    # get the frequency of each character
    freq = {}
    for c in legal_chars:
        freq[c] = data.count(c)
    # get the total number of characters
    total = sum(freq.values())
    # get the probability of each character
    for c in freq:
        freq[c] /= total
    
    # make sure this is a probability distribution
    assert abs(sum(freq.values()) - 1.0) < 1e-6

    return freq


def get_transition_matrix(data):
    """
    get the transition matrix from input data
    args:
        data: a long string
    return:
        a matrix Q indexing the transition probabilities
        Q(x, y) is the probability of transitioning from x to y
    """
    Q = np.zeros((len(legal_chars), len(legal_chars)))
    # count tuples
    for i in range(len(data) - 1):
        x = chars_to_index[data[i]]
        y = chars_to_index[data[i+1]]
        Q[x, y] += 1
    # normalize
    Q = Q / Q.sum(axis=1, keepdims=True)
    # make sure each row is a probability distribution
    assert np.isclose(Q.sum(axis=1), 1.0).all()

    return Q


def visualize(freq, Q):
    """
    visualize the one-point frequencies and the transition matrix Q
    """
    plt.imshow(np.array(list(freq.values()), dtype=np.float32).reshape((1, -1)), cmap='Blues')
    plt.colorbar(orientation="horizontal")
    plt.xticks(range(len(freq)), list(freq.keys()))
    plt.yticks([], [])
    plt.show()

    plt.imshow(Q, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(len(freq)), list(freq.keys()))
    plt.yticks(range(len(freq)), list(freq.keys()))
    plt.show()
    plt.close()


if __name__ == '__main__':
    # for debugging purposes
    data = load_data()
    print(len(data))

    freq = get_one_point_statistics(data)
    print(freq)
    print(len(freq))

    Q = get_transition_matrix(data)
    print(Q)
    print(Q.shape)

    visualize(freq, Q)
