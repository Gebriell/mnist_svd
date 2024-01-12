import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, norm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# a function to open the data file and read the data
def read_data(train_set, test_set, train_labels, test_labels):
    ''' open the data file and read the data'''

    # read the data into a list of lists
    with open(train_set, 'r') as f:
        data = [line.strip().split(',') for line in f.readlines()]

    with open(test_set, 'r') as f:
        test = [line.strip().split(',') for line in f.readlines()]

    with open(train_labels, 'r') as f:
        labels = [line.strip().split(',') for line in f.readlines()]

    with open(test_labels, 'r') as f:
        test_labels = [line.strip().split(',') for line in f.readlines()]

    # convert the data into a numpy array
    data = np.array(data, dtype=float)
    test = np.array(test, dtype=float)
    labels = np.array(labels, dtype=int)
    test_labels = np.array(test_labels, dtype=int)

    # split the array into 10 subarrays of size 400x400
    digit = np.split(data, 10)

    # create 3 dictionaries to store the left singular vectors, singular values, and right singular vectors
    left_singular_vectors = {}
    singular_values = {}
    right_singular_vectors = {}

    # compute the SVD for each subarray and store the results in the dictionaries
    for i in range(10):
        left_singular_vectors[i], singular_values[i], right_singular_vectors[i] = svd(digit[i], full_matrices=False)
        left_singular_vectors.update({i: left_singular_vectors[i]})
        singular_values.update({i: singular_values[i]})
        right_singular_vectors.update({i: right_singular_vectors[i]})

    # plot the original image and the reconstructed image from the 50 basis vectors
    basis = 50
    plt.figure(figsize=(9, 6)) # set the figure size to 9x6
    basis_vectors = np.concatenate([left_singular_vectors[x][:, :basis] for x in range(10)], axis=1) # 400x200 matrix of basis vectors
    plt.subplot(1, 2, 1) # plot the original image on the left
    plt.imshow(digit[3][0].reshape(20, 20), cmap='gray') # reshape the image to 20x20
    plt.title('Original') # set the title
    plt.subplot(1, 2, 2) # plot the reconstructed image on the right
    plt.imshow(basis_vectors.dot(basis_vectors.T).dot(digit[3][0]).reshape(20, 20), cmap='gray') # dot product of the basis vectors and the original image
    plt.title('Reconstructed with: ' + str(basis) + ' basis vectors')
    plt.show()

    # plot the singular values for the first subarray in log scale
    plt.figure(figsize=(10, 6)) # set the figure size to 10x5
    plt.plot(singular_values[0], color='orange', marker='o') 
    plt.yscale('log')
    plt.show()


def main():
    ''' main function '''
    read_data('handwriting_training_set.txt', 'handwriting_test_set.txt', 'handwriting_training_set_labels.txt', 'handwriting_test_set_labels.txt')


if __name__ == '__main__':
    main()

#.....................................................................................................................