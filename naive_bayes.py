# Dustin Le
# 1001130689

from sys import argv
import numpy as np

def naive_bayes(training_file, test_file):
    train = []
    test = []

    with open(training_file, 'r') as training:
        for line in training:
            train.append(line.split())
    
    with open(test_file, 'r') as testing:
        for line in testing:
            test.append(line.split())
    
    train = np.array(train).astype(np.float)
    test = np.array(test).astype(np.float)

    # ╭─━━━━━━━━━━━━━━─╮
    #   Training Phase
    # ╰─━━━━━━━━━━━━━━─╯

    rows = train.shape[0]
    columns = train.shape[1]
    # Number of rows of result array
    classes = np.unique(np.transpose(train)[columns-1]) 
    
    ans_sum = [[0 for x in range(columns)] for y in range(len(classes))]
    ans_sum = np.array(ans_sum).astype(np.float)

    count = [0 for x in range(len(classes))]

    # Mean of columns
    # Counter for class to keep array 0 based
    z = 0 
    
    # Sums up all columns based on class
    for k in range(int(min(classes)), int(max(classes)+1)):
        for i in range(rows):
            if train[i][columns-1] == k:
                # Keep count of class matches to do averaging
                count[z] += 1
                for j in range(columns-1):
                    ans_sum[z][j] += train[i][j]
        # Set last column as class number
        ans_sum[z][columns-1] = k
        z += 1
    
    # Mean all columns based on class
    ans_avg = ans_sum.copy()
    for i in range(len(ans_sum)):
        for j in range(columns-1):
            ans_avg[i][j] /= count[i]

    # Standard deviation of columns
    # Square difference
    temp = train.copy()
    z = 0
    for k in range(int(min(classes)), int(max(classes)+1)):
        for i in range(rows):
            if train[i][columns-1] == k:
                for j in range(columns-1):
                    # Subtracting mean and squaring result
                    temp[i][j] = (temp[i][j] - ans_avg[z][j])**2
        z += 1

    # Adding up square differences
    ans_std = [[0 for x in range(columns)] for y in range(len(classes))]
    ans_std = np.array(ans_std).astype(np.float)

    z = 0
    for k in range(int(min(classes)), int(max(classes)+1)):
        for i in range(rows):
            if train[i][columns-1] == k:
                for j in range(columns-1):
                    ans_std[z][j] += temp[i][j]
        # Set last column as class number
        ans_std[z][columns-1] = k
        z += 1


    # Getting mean of square difference and then rooting to get std
    for i in range(len(ans_std)):
        for j in range(columns-1):
            ans_std[i][j] /= count[i]
            if np.sqrt(ans_std[i][j]) < 0.01:
                ans_std[i][j] = 0.01
            else:
                ans_std[i][j] = np.sqrt(ans_std[i][j])
    
    # Output
    for i in range(ans_sum.shape[0]):
        for j in range(ans_sum.shape[1]-1):
            print('Class', str(int(ans_sum[i][ans_sum.shape[1]-1])) + ',', 'attribute', str(j+1) + ',', 'mean =', str('%.2f' % ans_avg[i][j]) + ',', 'std =', '%.2f' % ans_std[i][j])

    # ╭─━━━━━━━━━━━━━─╮
    #   Testing Phase
    # ╰─━━━━━━━━━━━━━─╯

    rows = test.shape[0]
    columns = test.shape[1]
    count = [0 for x in range(len(classes))]
    count = np.array(count)
    classes = np.unique(np.transpose(test)[columns-1]) 

    # One-dimension Gaussians
    temp = test.copy()
    z = 0
    for k in range(int(min(classes)), int(max(classes)+1)):
        for i in range(rows):
            if test[i][columns-1] == k:
                for j in range(columns-1):
                    temp[i][j] = (1/(ans_std[z][j]*np.sqrt(2*np.pi))*np.e**(-((test[i][j]-ans_avg[z][j])**2)/(2*ans_std[z][j]**2)))
        z += 1

    # The product of the gaussians of every row
    gaussian = [[1 for y in range(2)] for x in range(rows)]
    gaussian = np.array(gaussian).astype(np.float)

    for i in range(rows):
        for j in range(columns-1):
            gaussian[i] *= temp[i][j]
        gaussian[i][1] = temp[i][-1]
        x = temp[i][-1] - 1
        count[int(x)] += 1

    # Bayes rule
    p_C = count/sum(count)
    
    top = gaussian.copy()
    bottom = 0
    ans_classifier = gaussian.copy()
    ans_classifier = np.array(ans_classifier).astype(np.float)
    
    # P(x | C) * p(C)
    for i in range(len(top)):
        j = int((top[i][1]) - 1)
        top[i][0] *= p_C[j]

    # P(x) = P(x | C1) * p(C1) + ...
    for i in range(len(gaussian)):
        j = int((gaussian[i][1]) - 1)
        bottom += gaussian[i][0]*p_C[j] 

    # P(C | x) = P(x | C) * p(C) / P(x)
    top[:, :-1] = top[:, :-1]/bottom
    ans_classifier = top

    # Output
    ans_accuracy = 0
    for i in range(rows):
        temp = [0 for x in range(len(classes))] 
        for j in range(rows):
            if int(test[i][-1]) == ans_classifier[j][1]:
                temp[(int(test[i][-1])-1)] += 1
        predicted = np.argmax(temp)
        ties = np.argwhere(temp == predicted)
        isin = np.isin(ties, test[i][-1])
        if len(ties) == 0 and predicted+1 == test[i][-1]:
            accuracy = 1
        elif len(ties) == 0 and predicted+1 != test[i][-1]:
            accuracy = 0
        elif len(ties) > 0 and isin.any():
            accuracy = 1/len(ties)
        else:
            accuracy = 0
        ans_accuracy += accuracy
        print('ID=' + str(i+1) + ',', 'predicted=' + str(predicted+1) + ',', 'probability =', str('%.4f' % ans_classifier[predicted][0]) + ',', 'true=' + str(int(test[i][-1])) + ',', 'accuracy=', '%4.2f' % accuracy, '\n')
    
    ans_accuracy /= rows
    print('classification accuracy=' + str('%6.4f' % ans_accuracy))

naive_bayes(argv[1], argv[2])