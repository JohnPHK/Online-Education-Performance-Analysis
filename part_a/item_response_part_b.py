# from starter_code.utils import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    c = data["is_correct"]
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]

        theta_i = theta[cur_user_id]
        beta_j = beta[cur_question_id]
        alpha_j = alpha[cur_question_id]

        log_lklihood += (c[i] * np.log(sigmoid((alpha_j * (theta_i - beta_j)).sum())) +
                         (1 - c[i]) * np.log(1 - sigmoid((alpha_j * (theta_i - beta_j)).sum())))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    M, N = len(set(data["user_id"])), len(set(data["question_id"]))

    dl_dtheta = np.zeros((M, 1))
    dl_dbeta = np.zeros((N, 1))

    c = data["is_correct"]
    # NOTE: gradient ascent for maximizing the LLK
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        theta_i = theta[cur_user_id]
        beta_j = beta[cur_question_id]
        alpha_j = alpha[cur_question_id]
        dl_dtheta[cur_user_id] += alpha_j * (c[i] - sigmoid(alpha_j * (theta_i - beta_j)))

    theta = theta + (lr * dl_dtheta)

    for j in range(len(data["user_id"])):
        cur_user_id = data["user_id"][j]
        cur_question_id = data["question_id"][j]
        theta_i = theta[cur_user_id]
        beta_j = beta[cur_question_id]
        alpha_j = alpha[cur_question_id]
        dl_dbeta[cur_question_id] += alpha_j * (sigmoid(alpha_j * (theta_i - beta_j)) - c[j])

    beta = beta + (lr * dl_dbeta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, val_data, lr, iterations, alpha):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    M, N = len(set(train_data["user_id"])), len(set(train_data["question_id"]))

    theta = np.zeros((M, 1))
    beta = np.zeros((N, 1))

    val_acc_lst = []
    train_acc_lst = []
    train_llds = []
    val_llds = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta, alpha=alpha)
        train_llds.append(-1 * neg_lld)

        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha)
        val_llds.append(-1 * val_neg_lld)

        train_score = evaluate(data=train_data, theta=theta, beta=beta, alpha=alpha)
        val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)

        train_acc_lst.append(train_score)
        val_acc_lst.append(val_score)
        print("NLLK: {} \t Score: {}".format(neg_lld, val_score))
        theta, beta = update_theta_beta(train_data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_llds, val_llds


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        # x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot_llds(iters, llds, dataset, title):
    labels = {"title": "{} - {} Log Likelihoods Vs. Iterations".format(title, dataset),
              "x": "Iterations",
              "y": "Log Likelihood"}

    plt.title(labels["title"])
    plt.xlabel(labels["x"])
    plt.ylabel(labels["y"])
    plt.plot(list(range(iters)), llds)
    plt.show()


def plot_accuracies(iters, train_acc, val_acc, title):
    plt.title("{} - Accuracy VS. Iteration".format(title))
    plt.plot(list(range(iters)), train_acc, label="Training")
    plt.plot(list(range(iters)), val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def compute_test_score_per_user(data):
    """Calculate the ratio of the number of questions a
    student answered correctly to the number of questions they attempted
    """

    M = len(set(data["user_id"]))
    user_num_questions_correct = np.zeros((M, 1))
    user_num_questions_attempt = np.zeros((M, 1))

    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        correct = data["is_correct"][i]

        user_num_questions_correct[cur_user_id] += correct
        user_num_questions_attempt[cur_user_id] += 1

    test_score_per_user = user_num_questions_correct / user_num_questions_attempt
    return test_score_per_user


def compute_discrimination_indices(sm, test_scores):
    """
    Reference: https://www.theclassroom.com/how-does-a-grading-curve-work-12146346.html
    """
    M, N = sm.shape
    alpha = np.zeros((N, 1))

    sm_with_scores = np.zeros((M, N+1))
    sm_with_scores[:M, :N] = sm.todense()
    sm_with_scores[:, N] = test_scores.T

    sorted_sm = sm_with_scores[np.argsort(sm_with_scores[:, N])]
    sorted_sm = sorted_sm[:M, :N]

    non_entries = sum(np.isnan(sorted_sm))
    sorted_sm[np.isnan(sorted_sm)] = 0.0

    groups = np.split(sorted_sm, 2)
    low_score, high_score = groups[0], groups[1]

    for i in range(high_score.shape[1]):
        h = high_score[:, i].sum()
        l = low_score[:, i].sum()
        t = (sorted_sm.shape[0] - non_entries[i])/2
        alpha[i] = (h - l) / t
    return alpha



def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    train_sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # pre-process: compute discrimination indices for each question
    test_scores = compute_test_score_per_user(train_data)
    alpha = compute_discrimination_indices(train_sparse_matrix, test_scores)

    hyperparameters = {"lr": 0.1, "iters": 100}
    #
    accs = []
    for lr in [hyperparameters["lr"]]:
        print(f'Learning Rate: {lr}')
        theta, beta, train_acc_lst, val_acc_lst, train_llds, val_llds = \
            irt(train_data,
                val_data,
                lr=lr,
                iterations=hyperparameters["iters"],
                alpha=alpha)

        plot_accuracies(hyperparameters["iters"],
                        train_acc_lst,
                        val_acc_lst,
                        title=f'LR: {lr}')

        plot_llds(hyperparameters["iters"],
                  train_llds,
                  "Training",
                  title=f'LR: {lr}')
        plot_llds(hyperparameters["iters"],
                  val_llds,
                  "Validation",
                  title=f'LR: {lr}')

        # Part (c)
        val_acc = evaluate(val_data, theta, beta, alpha)
        test_acc = evaluate(test_data, theta, beta, alpha)
        print(f"Final Validation Accuracy: {val_acc}")
        print(f"Final Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    thetas = np.linspace(-5, 5)

    question_ids = np.random.choice(train_data["question_id"], 5)
    question_betas = [beta[idx] for idx in question_ids]
    question_alphas = [alpha[idx] for idx in question_ids]

    for i in range(len(question_ids)):
        plt.plot(thetas, sigmoid(question_alphas[i] * (thetas - question_betas[i])), label=f"Question {question_ids[i]}")
    plt.legend()
    plt.title("Probability Correct Response VS. Student Ability")
    plt.xlabel("Student Ability (Theta)")
    plt.ylabel("Correct Response")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
