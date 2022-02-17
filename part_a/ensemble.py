# TODO: complete this file.
from utils import *
from sklearn.utils import resample


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def ir_neg_log_likelihood(data, theta, beta):
    log_lklihood = 0.
    c = data["is_correct"]
    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        log_lklihood += (c[i] * np.log(sigmoid((theta[cur_user_id] - beta[cur_question_id]).sum())) +
                         (1 - c[i]) * np.log(1 - sigmoid((theta[cur_user_id] - beta[cur_question_id]).sum())))
    return -log_lklihood

def item_response_evaluate(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def ir_update_theta_beta(data, lr, theta, beta):
    M, N = len(set(data["user_id"])), len(set(data["question_id"]))

    dl_dtheta = np.zeros((M, 1))
    dl_dbeta = np.zeros((N, 1))

    c = data["is_correct"]

    for i in range(len(data["user_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        dl_dtheta[cur_user_id] += c[i] - sigmoid(theta[cur_user_id] - beta[cur_question_id])

    theta = theta + (lr * dl_dtheta)

    for j in range(len(data["user_id"])):
        cur_user_id = data["user_id"][j]
        cur_question_id = data["question_id"][j]
        dl_dbeta[cur_question_id] += sigmoid(theta[cur_user_id] - beta[cur_question_id]) - c[j]

    beta = beta + (lr * dl_dbeta)
    return theta, beta


def bag_irt(train_data, val_data, lr, iterations):
    M, N = len(set(train_data["user_id"])), len(set(train_data["question_id"]))

    theta = np.zeros((M, 1))
    beta = np.zeros((N, 1))

    val_acc_lst = []
    train_acc_lst = []
    train_llds = []
    val_llds = []

    for i in range(iterations):
        neg_lld = ir_neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_llds.append(-1 * neg_lld)

        val_neg_lld = ir_neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llds.append(-1 * val_neg_lld)

        train_score = item_response_evaluate(data=train_data, theta=theta, beta=beta)
        val_score = item_response_evaluate(data=val_data, theta=theta, beta=beta)

        train_acc_lst.append(train_score)
        val_acc_lst.append(val_score)
        theta, beta = ir_update_theta_beta(train_data, lr, theta, beta)

    return theta, beta, train_acc_lst, val_acc_lst, train_llds, val_llds

def bag_evaluate(data, thetas, betas):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x_1 = (thetas[0][u] - betas[0][q]).sum()
        x_2 = (thetas[1][u] - betas[1][q]).sum()
        x_3 = (thetas[2][u] - betas[2][q]).sum()
        p_a_1 = sigmoid(x_1)
        p_a_2 = sigmoid(x_2)
        p_a_3 = sigmoid(x_3)
        pred.append(np.mean([p_a_1, p_a_2, p_a_3]) >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    N = len(train_data['user_id'])
    to_bootstrap = [train_data['user_id'], train_data['question_id'], train_data['is_correct']]
    to_bootstrap = np.array(to_bootstrap)
    to_bootstrap = np.transpose(to_bootstrap)

    thetas = []
    betas = []

    for _ in range(3):
        resampled_data = resample(to_bootstrap, n_samples=N, replace=True)
        resampled_data_T = np.transpose(resampled_data)

        resampled_train = dict()
        resampled_train['user_id'] = resampled_data_T[0]
        resampled_train['question_id'] = resampled_data_T[1]
        resampled_train['is_correct'] = resampled_data_T[2]

        hyperparameters = {"lr": 0.01, "iters": 50}

        for lr in [hyperparameters["lr"]]:
            theta, beta, train_acc_lst, val_acc_lst, train_llds, val_llds = \
                bag_irt(resampled_train, 
                                  val_data, 
                                  lr=lr, 
                                  iterations=hyperparameters["iters"])
        thetas.append(theta)
        betas.append(beta)

    val_acc = bag_evaluate(val_data, thetas, betas)
    test_acc = bag_evaluate(test_data, thetas, betas)

    print("Bagged Validation Accuracy: {}".format(val_acc))
    print("Bagged Test Accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()
