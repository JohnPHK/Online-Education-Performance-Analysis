from sklearn.impute import KNNImputer
from sklearn import neighbors
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(np.transpose(matrix))

    total_prediction = 0
    total_accurate = 0
    threshold = 0.5
    for i in range(len(valid_data["is_correct"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        if mat[cur_question_id, cur_user_id] >= threshold and valid_data["is_correct"][i]:
            total_accurate += 1
        if mat[cur_question_id, cur_user_id] < threshold and not valid_data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    acc = total_accurate / float(total_prediction)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv('../data')

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    impute_by_user_accuracies = []
    impute_by_item_accuracies = []

    best_k_user = 0
    best_accuracy_user = 0
    print("knn_impute_by_user:")
    for k in k_values:
        print("k:", k)
        acc = knn_impute_by_user(sparse_matrix.copy(), val_data, k)
        impute_by_user_accuracies.append(acc)
        if acc > best_accuracy_user:
            best_accuracy_user = acc
            best_k_user = k
    print("The best k:", best_k_user)
    print("The best accuracy:", best_accuracy_user)

    # Generate a plot
    plt.plot(k_values, impute_by_user_accuracies, color='red')
    plt.title("Plot for knn_impute_by_user")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.savefig("Q1_user.png")
    plt.show()

    best_k_item = 0
    best_accuracy_item = 0
    print("knn_impute_by_item:")
    for k in k_values:
        print("k:", k)
        acc = knn_impute_by_item(sparse_matrix.copy(), val_data, k)
        impute_by_item_accuracies.append(acc)
        if acc > best_accuracy_item:
            best_accuracy_item = acc
            best_k_item = k
    print("The best k:", best_k_item)
    print("The best accuracy:", best_accuracy_item)

    # Generate a plot
    plt.plot(k_values, impute_by_item_accuracies, color='red')
    plt.title("Plot for knn_impute_by_item")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.savefig("Q1_item.png")
    plt.show()

    # Below is the comparison between user-based and item-based
    #user-basd:
    nbrs = KNNImputer(n_neighbors=best_k_user)
    print(best_k_user)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(sparse_matrix.copy())
    user_test_acc = sparse_matrix_evaluate(test_data, mat)

    #item-based:
    nbrs = KNNImputer(n_neighbors=best_k_item)
    print(best_k_item)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(np.transpose(sparse_matrix.copy()))

    total_prediction = 0
    total_accurate = 0
    threshold = 0.5
    for i in range(len(test_data["is_correct"])):
        cur_user_id = test_data["user_id"][i]
        cur_question_id = test_data["question_id"][i]
        if mat[cur_question_id, cur_user_id] >= threshold and test_data["is_correct"][i]:
            total_accurate += 1
        if mat[cur_question_id, cur_user_id] < threshold and not test_data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    item_test_acc = total_accurate / float(total_prediction)

    print("User-based. The accuracy on test data: {}".format(user_test_acc))
    print("Item-based. The accuracy on test data: {}".format(item_test_acc))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
