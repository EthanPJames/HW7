import numpy as np
import matplotlib.pyplot as plt

# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))
        
    # iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.
    
    for d in degrees:
        x1 = feature_matrix(x,d) #should it be capital X
        w = least_squares(x1,y)  #Should x1 be a capital X
        #paramFits.append(w.tolist())
        paramFits.append(w)

    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[x_i**j for j in range(d,-1,-1)] for x_i in x]
    return X


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    #X_pinv = np.linalg.pinv(X) #Calculate inverse of X
    B = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X),y))  #Calc least squares regression using matrix multipilication
    return B


if __name__ == "__main__":
    datapath = "poly.txt"

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))
###########################################   RESUME HERE   ###################################################
    # Problem 1.
    # Complete 'main, 'feature_matrix', and 'least_squares' functions above
    #DONE

    # Problem 2.
    ## degrees 2 and 4 have been provided as test cases. The output should match that as specified on the README.

    ## Update the degrees,d to include 1, 3, 5 and 6. i.e. [1,2,3,4,5,6] and
    # Write out the resulting estimated functions for each d.
    degrees = [1,2,3, 4,5,6]
    paramFits = main(datapath, degrees)
    # print(paramFits)
    for idx in range(len(degrees)):
        print("y_hat(x_"+str(degrees[idx])+")")
        print(paramFits[idx])
        print("****************")

    # Problem 3.
    # Visualize the dataset and these fitted models on a single graph
    # Use the 'scatter' and 'plot' functions in the `matplotlib.pyplot` module.
    # Draw a scatter plot
    plt.scatter(x, y, color='black', label='data')
    x.sort()

    for params in paramFits:
        ### Fill in your code ###
        d = len(params) - 1
        X = feature_matrix(x,d)
        X = np.array(X)
        y_predicted = np.dot(X, params)
        plt.plot(x, y_predicted, label='d = ' + str(d), linewidth=2, alpha=0.5)

        
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.legend(fontsize=10, loc='upper left')

    plt.show()

    



    ## Problem 4

    # when x = 2; what is the predicted output
    '''
    fill in your code
    '''

    
    #Udpated the deegrees here
    degrees = [1,2,3,4,5,6]
    paramFits = main(datapath, degrees)
    #Print it
    for idx in range(len(degrees)):
        #Create the print statemtns for the different degrees
        print("y_hat(x_"+str(degrees[idx])+")")
        print(paramFits[idx])
        arr_coef = np.array([2**b for b in range(degrees[idx], -1, -1)])
        print("Coefficeints = ", arr_coef)
        #set the value
        value = np.dot(paramFits[idx], arr_coef)
        #Print out the value
        print("Value = ", value)
        print("**************************")
        plt.show() #may be unnecessary



    #Now find out when x = 2
    y = paramFits[2][0]*2**3 + paramFits[2][1]*2**2 + paramFits[2][2]*2 + paramFits[2][3]
    print("Y' is= ", y) #Is this the correct way to print out?????????????????????????????????????????????!!!!!!!!!!!!!!!!!!!!!


