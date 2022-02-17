from utils import *
from scipy.linalg import sqrtm

import numpy as np

import matplotlib.pyplot as plt     # we imported this 


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
        
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    
    ###################     MY CODE     ########################
    
    #u[n] = u[n] + lr*(c - u[n].transpose()*z[q])*z[q] 
    
    u[n] = u[n] + np.dot((lr*(c - np.dot(np.transpose(u[n]), z[q]))), z[q]) 
    
    
    
    #z[q] = z[q] + lr*(c - u[n].transpose()*z[q])*u[n]
    
    z[q] = z[q] + np.dot((lr*(c - np.dot(np.transpose(u[n]), z[q]))), u[n])
    
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    mat = None
    
    ###################     MY CODE     ########################
    
    # we call update_u_z in a loop and compute the squared_error_loss()  
    # each time, and if it is small enough, we are done 
    
                
    for i in range(num_iteration):
        u,z = update_u_z(train_data, lr, u, z)
        
    print("squared_error_loss for k =", k, " on train_data is: ", squared_error_loss(train_data, u, z)) 
    # print("squared_error_loss for k =", k, " on val_data is: ", squared_error_loss(val_data, u, z)) 
    # print("squared_error_loss for k =", k, " on test_data is: ", squared_error_loss(test_data, u, z)) 
    print("\n") 
     
    mat = np.dot(u, np.transpose(z))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat



def als_modified(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    mat = None
    
                
    for i in range(num_iteration):
        u,z = update_u_z(train_data, lr, u, z)
        
    #print("squared_error_loss for k =", k, " is: ", squared_error_loss(train_data, u, z)) 
    #print("\n") 
     
    mat = np.dot(u, np.transpose(z))
    
    
    return mat, u, z 




def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    
   

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    
    # my code 
    '''
    for a in train_data:
        print(a)
        for b in a:
            print(b)
            
    print(train_data)
    #print(train_matrix)
    #
    '''
    
    k_1 = svd_reconstruct(train_matrix, 1)
    evaluate_1 = sparse_matrix_evaluate(train_data, k_1)
    print("(svd with k=1) sparse_matrix_evaluate function gives accuracy   :", evaluate_1)
    
    predictions = sparse_matrix_predictions(val_data, k_1)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 1 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_1)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 1 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_3 = svd_reconstruct(train_matrix, 3)
    evaluate_3 = sparse_matrix_evaluate(train_data, k_3)
    print("(svd with k=3) sparse_matrix_evaluate function gives accuracy   :", evaluate_3)
    
    predictions = sparse_matrix_predictions(val_data, k_3)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 3 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_3)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 3 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_5 = svd_reconstruct(train_matrix, 5)
    evaluate_5 = sparse_matrix_evaluate(train_data, k_5)
    print("(svd with k=5) sparse_matrix_evaluate function gives accuracy   :", evaluate_5)
    
    predictions = sparse_matrix_predictions(val_data, k_5)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 5 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_5)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 5 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_8 = svd_reconstruct(train_matrix, 8)
    evaluate_8 = sparse_matrix_evaluate(train_data, k_8)
    print("(svd with k=8) sparse_matrix_evaluate function gives accuracy   :", evaluate_8)
    
    predictions = sparse_matrix_predictions(val_data, k_8)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 8 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_8)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 8 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_12 = svd_reconstruct(train_matrix, 12)
    evaluate_12 = sparse_matrix_evaluate(train_data, k_12)
    print("(svd with k=12) sparse_matrix_evaluate function gives accuracy  :", evaluate_12)
    
    predictions = sparse_matrix_predictions(val_data, k_12)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 12 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_12)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 12 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_30 = svd_reconstruct(train_matrix, 30)
    evaluate_30 = sparse_matrix_evaluate(train_data, k_30)
    print("(svd with k=30) sparse_matrix_evaluate function gives accuracy :", evaluate_30)
    
    predictions = sparse_matrix_predictions(val_data, k_30)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 30 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_30)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 30 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_50 = svd_reconstruct(train_matrix, 50)
    evaluate_50 = sparse_matrix_evaluate(train_data, k_50)
    print("(svd with k=50) sparse_matrix_evaluate function gives accuracy :", evaluate_50)
    
    predictions = sparse_matrix_predictions(val_data, k_50)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 50 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_50)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 50 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_80 = svd_reconstruct(train_matrix, 80)
    evaluate_80 = sparse_matrix_evaluate(train_data, k_80)
    print("(svd with k=80) sparse_matrix_evaluate function gives accuracy :", evaluate_80)
    
    predictions = sparse_matrix_predictions(val_data, k_80)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 80 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_80)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 80 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    k_100 = svd_reconstruct(train_matrix, 100)
    evaluate_100 = sparse_matrix_evaluate(train_data, k_100)
    print("(svd with k=100) sparse_matrix_evaluate function gives accuracy :", evaluate_100)
    
    predictions = sparse_matrix_predictions(val_data, k_100)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 100 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_100)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 100 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    
    k_150 = svd_reconstruct(train_matrix, 150)
    evaluate_150 = sparse_matrix_evaluate(train_data, k_150)
    print("(svd with k=150) sparse_matrix_evaluate function gives accuracy :", evaluate_150)
    
    predictions = sparse_matrix_predictions(val_data, k_150)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 150 we get accuracy: ", accuracy)
     
    predictions = sparse_matrix_predictions(test_data, k_150)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 150 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    
    k_200 = svd_reconstruct(train_matrix, 200)
    evaluate_200 = sparse_matrix_evaluate(train_data, k_200)
    print("(svd with k=200) sparse_matrix_evaluate function gives accuracy :", evaluate_200)
    
    predictions = sparse_matrix_predictions(val_data, k_200)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 200 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, k_200)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 200 we get accuracy: ", accuracy)
    print("\n")
    
    
    
    
    k_500 = svd_reconstruct(train_matrix, 500)
    evaluate_500 = sparse_matrix_evaluate(train_data, k_500)
    print("(svd with k=500) sparse_matrix_evaluate function gives accuracy :", evaluate_500)
    
    predictions = sparse_matrix_predictions(val_data, k_500)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 500 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, k_500)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 500 we get accuracy: ", accuracy)
    
    print("\n") 
    
    
    
    
    
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    print("We now move on to the Matrix Factorization and ALS part.\n\n")
    
    #reconstructed_matrix = als(train_data, k=100, lr=0.001, num_iteration=10000000)   takes 6 minutes 
    reconstructed_matrix = als(train_data, k=5, lr=0.3, num_iteration=100000)
    print("our reconstructed matrix is: ")
    print(reconstructed_matrix) 
    print("\n") 
    evaluate_recons_mat = sparse_matrix_evaluate(train_data, reconstructed_matrix)
    print("sparse_matrix_evaluate function gives accuracy: ", evaluate_recons_mat)
    
    predictions = sparse_matrix_predictions(val_data, reconstructed_matrix)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 5 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, reconstructed_matrix)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 5 we get accuracy: ", accuracy)
    print("\n") 
    
    
    
    
    reconstructed_matrix = als(train_data, k=100, lr=0.3, num_iteration=100000)
    print("our reconstructed matrix is: ")
    print(reconstructed_matrix) 
    print("\n") 
    evaluate_recons_mat = sparse_matrix_evaluate(train_data, reconstructed_matrix)
    print("sparse_matrix_evaluate function gives accuracy: ", evaluate_recons_mat)
    print("\n") 
    
    predictions = sparse_matrix_predictions(val_data, reconstructed_matrix)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 100 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, reconstructed_matrix)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 100 we get accuracy: ", accuracy)
    print("\n") 
    
    
    
    
    reconstructed_matrix = als(train_data, k=150, lr=0.3, num_iteration=100000)
    print("our reconstructed matrix is: ")
    print(reconstructed_matrix) 
    print("\n") 
    evaluate_recons_mat = sparse_matrix_evaluate(train_data, reconstructed_matrix)
    print("sparse_matrix_evaluate function gives accuracy: ", evaluate_recons_mat)
    print("\n") 
    
    predictions = sparse_matrix_predictions(val_data, reconstructed_matrix)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 150 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, reconstructed_matrix)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 150 we get accuracy: ", accuracy)
    print("\n") 
    
    
    
    
    reconstructed_matrix = als(train_data, k=200, lr=0.3, num_iteration=100000)
    print("our reconstructed matrix is: ")
    print(reconstructed_matrix) 
    print("\n") 
    evaluate_recons_mat = sparse_matrix_evaluate(train_data, reconstructed_matrix)
    print("sparse_matrix_evaluate function gives accuracy: ", evaluate_recons_mat)
    print("\n") 
    
    predictions = sparse_matrix_predictions(val_data, reconstructed_matrix)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 200 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, reconstructed_matrix)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 200 we get accuracy: ", accuracy)
    print("\n") 
    
    
    
    
    reconstructed_matrix = als(train_data, k=500, lr=0.3, num_iteration=100000)
    print("our reconstructed matrix is: ")
    print(reconstructed_matrix) 
    print("\n") 
    evaluate_recons_mat = sparse_matrix_evaluate(train_data, reconstructed_matrix)
    print("sparse_matrix_evaluate function gives accuracy: ", evaluate_recons_mat)
    print("\n") 
    
    predictions = sparse_matrix_predictions(val_data, reconstructed_matrix)
    accuracy = evaluate(val_data, predictions)
    print("on the validation set with k = 500 we get accuracy: ", accuracy)
    
    predictions = sparse_matrix_predictions(test_data, reconstructed_matrix)
    accuracy = evaluate(test_data, predictions)
    print("on the test set with k = 500 we get accuracy: ", accuracy)
    print("\n") 
    
    # we now do part (e)  
    ########################### (E) #############################
    
    

    
    
    num_iter_list = [] 
    list_train_err= [] 
    
    list_val_acc = [] 
    list_val_err = [] 
    
    list_test_acc = [] 
    list_val_test = [] 
    
    i = 50 
    
    '''
    while i < 1000000:
        num_iter_list.append(i) 
        recons_mat, u, z = als_modified(train_data, k=200, lr=0.3, num_iteration=i) 
        
        list_train_err.append(squared_error_loss(train_data, u, z))
        
        predictions = sparse_matrix_predictions(val_data, recons_mat)
        accuracy = evaluate(val_data, predictions) 
        list_val_acc.append(accuracy)
        list_val_err.append(squared_error_loss(val_data, u, z))
        
        predictions = sparse_matrix_predictions(test_data, recons_mat)
        accuracy = evaluate(test_data, predictions)
        list_test_acc.append(accuracy)
        list_val_test.append(squared_error_loss(test_data, u, z)) 
        
        if i < 100000:
            i = i + 1000 
        else:
            i = i + 200000
    '''
    # to get a graph with more iterations, comment out the below loop, and 
    # uncomment the above loop 
            
    while i < 100000:
        num_iter_list.append(i) 
        recons_mat, u, z = als_modified(train_data, k=200, lr=0.3, num_iteration=i) 
        
        list_train_err.append(squared_error_loss(train_data, u, z))
        
        predictions = sparse_matrix_predictions(val_data, recons_mat)
        accuracy = evaluate(val_data, predictions) 
        list_val_acc.append(accuracy)
        list_val_err.append(squared_error_loss(val_data, u, z))
        
        predictions = sparse_matrix_predictions(test_data, recons_mat)
        accuracy = evaluate(test_data, predictions)
        list_test_acc.append(accuracy)
        list_val_test.append(squared_error_loss(test_data, u, z)) 
        
        i = i + 500
    
        
   
        
    x1 = num_iter_list 
    y1 = list_train_err 
    plt.plot(x1, y1, label = "train squared error") 
    
    x2 = num_iter_list 
    y2 = list_val_err 
    plt.plot(x2, y2, label = "validation squared error") 
    
    
    plt.xlabel('number of iterations') 
    plt.ylabel('squared error') 
    plt.title("Squared Error vs Number of Iterations") 
    #plt.title("Squared Error vs Number of Iterations(1000000 iterations)") 
    
    plt.legend() 
    plt.show(block=False) 
    
    
    
    
    
    x1 = num_iter_list 
    y1 = list_val_acc
    plt.plot(x1, y1, label = "validation accuracy") 
    
    x2 = num_iter_list 
    y2 = list_test_acc 
    plt.plot(x2, y2, label = "test accuracy") 
    
    
    plt.xlabel('number of iterations') 
    plt.ylabel('accuracy') 
    plt.title("Accuracy vs Number of Iterations") 
    #plt.title("Accuracy vs Number of Iterations(1000000 iterations)") 
    
    plt.legend() 
    plt.show() 
    
    # lastly we report the final validation and test accuracy 
    print("final validation accuracy is: ", list_val_acc[-1]) 
    print("final test accuracy is: ", list_test_acc[-1]) 
    #############################################################
    
    pass

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
