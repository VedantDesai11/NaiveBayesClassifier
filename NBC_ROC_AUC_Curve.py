import numpy as np
import matplotlib.pyplot as plt
import math

# Generate Train and Test data for label 0
class dataSetLabel0():

    def __init__(self, mu, sigma, number):
        self.mu = mu
        self.sigma = sigma
        self.number = number

    def generateData(self):

        mu = self.mu
        sigma = self.sigma
        number = self.number

        #training data
        x1, y1 = np.random.multivariate_normal(mu, sigma, number).T
        trainData = []
        trainLabel = []
        for x in range(len(x1)):
            trainData.append([x1[x], y1[x]])
            trainLabel.append(0)

        trainData = np.array(trainData)
        trainLabel = np.array(trainLabel)

        #testing data
        x_test, y_test = np.random.multivariate_normal(mu, sigma, number).T
        testData = []
        testLabel = []
        for x in range(len(x_test)):
            testData.append([x_test[x], y_test[x]])
            testLabel.append(0)

        testData = np.array(testData)
        testLabel = np.array(testLabel)
        return trainData, trainLabel, testData, testLabel

# Generate Train and Test data for label 1
class dataSetLabel1():

    def __init__(self, mu, sigma, number):
        self.mu = mu
        self.sigma = sigma
        self.number = number

    def generateData(self):

        mu = self.mu
        sigma = self.sigma
        number = self.number

        #training data
        x1, y1 = np.random.multivariate_normal(mu, sigma, number).T
        trainData = []
        trainLabel = []
        for x in range(len(x1)):
            trainData.append([x1[x], y1[x]])
            trainLabel.append(1)

        trainData = np.array(trainData)
        trainLabel = np.array(trainLabel)

        #testing data
        x_test, y_test = np.random.multivariate_normal(mu, sigma, number).T
        testData = []
        testLabel = []
        for x in range(len(x_test)):
            testData.append([x_test[x], y_test[x]])
            testLabel.append(1)

        testData = np.array(testData)
        testLabel = np.array(testLabel)
        return trainData, trainLabel, testData, testLabel


def myNB(trainData, trainLabel, testData, testLabel):

    numberOfTrainData = len(trainLabel)
    numberOfTestData = len(testData)

    seperated_trainData, meanSDCount_trainData = preprocess_data(trainData, trainLabel)

    prediction = []
    probabilties = []
    error = 0

    for data in testData:
        probabilty = {}
        p = []
        for key, value in meanSDCount_trainData.items():

            #probabiltyOfLabel
            probabilty[key] = value[0][2]/numberOfTestData

            for i in range(len(data)):
                probabilty[key] *= calculateProbabilty(data[i], value[i][0], value[i][1])

        highest = 0
        label = 0

        for x in probabilty.values():
            p.append(x)

        for key, value in probabilty.items():

            if value > highest:
                label = key
                highest = value

        prediction.append(label)
        probabilties.append(highest)

    for x in range(len(testLabel)):
        if testLabel[x] != prediction[x]:
            error += 1
        #print(testLabel[x], prediction[x], probabilties[x])

    return prediction, probabilties, (error/numberOfTestData)


def preprocess_data(dataset, labels):

    classes = list(np.unique(labels))
    seperatedData = seperate_by_class(dataset, labels)
    meanSDCount_of_data = {}

    for key in seperatedData.keys():

        if key not in meanSDCount_of_data:
            meanSDCount_of_data[key] = []

        for i in range(len(seperatedData[key][0])):
            meanSDCount_of_data[key].append(calculateMeanSdCount(seperatedData[key][:, i]))

    return seperatedData, meanSDCount_of_data


def seperate_by_class(dataset, labels):

    seperatedData = {}
    for index, data in enumerate(dataset):

        label = labels[index]
        if label not in seperatedData:
            seperatedData[label] = []

        seperatedData[label].append(data)

    for keys in seperatedData.keys():
        seperatedData[keys] = np.array(seperatedData[keys])

    return seperatedData


def calculateMeanSdCount(values):
    mean = sum(values)/float(len(values))
    variance = sum([(x - mean) ** 2 for x in values]) / float(len(values) - 1)
    sd = math.sqrt(variance)
    return mean, sd, len(values)


def calculateProbabilty(data, mean, sigma):
    exponent = math.exp(-((data - mean) ** 2 / (2 * sigma ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * exponent


def calculateTRPandFPR(pred, testLabel, posterior, threshold = 1):

    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    for x in range(len(pred)):
        if pred[x] == 1 and testLabel[x] == 1 and posterior[x] < threshold:
            truePositive += 1
        if pred[x] == 0 and testLabel[x] == 0 and posterior[x] < threshold:
            trueNegative += 1
        if pred[x] == 1 and testLabel[x] == 0 and posterior[x] < threshold:
            falsePositive += 1
        if pred[x] == 0 and testLabel[x] == 1 and posterior[x] < threshold:
            falseNegative += 1

    matrix = [[truePositive,falsePositive], [falseNegative,trueNegative]]

    tpr = truePositive/(truePositive+falseNegative)
    fnr = 1 - trueNegative/(falsePositive+trueNegative)

    return tpr, fnr


def calculateAUC(x,y):

    area = 0
    for i in range(len(x)):
        if i != 0:
            area += (x[i] - x[i-1]) * y[i-1] + (0.5 * (x[i] - x[i-1]) * (y[i] - y[i-1]))

    return -area


def main():

    tpr = [1]
    fnr = [1]

    # EQUAL SAMPLES

    mu1 = [1, 0]
    sigma1 = [[1, .75], [0.75, 1]]
    mu2 = [0, 1]
    sigma2 = [[1, .75], [0.75, 1]]

    dataSet1 = dataSetLabel0(mu1, sigma1, 500)
    dataSet2 = dataSetLabel1(mu2, sigma2, 500)

    trainData0, trainLabel0, testData0, testLabel0 = dataSet1.generateData()
    trainData1, trainLabel1, testData1, testLabel1 = dataSet2.generateData()

    trainData = np.concatenate((trainData0, trainData1))
    testData = np.concatenate((testData0, testData1))
    trainLabel = np.concatenate((trainLabel0, trainLabel1))
    testLabel = np.concatenate((testLabel0, testLabel1))

    pred, posterior, err = myNB(trainData, trainLabel, testData, testLabel)
    maxProb = max(posterior)
    for x in range(2,52):
        tprate, fnrate = calculateTRPandFPR(pred, testLabel, posterior, maxProb/x)
        tpr.append(tprate)
        print(tprate, fnrate)
        fnr.append(fnrate)

    tpr.append(0)
    fnr.append(0)


    plt.plot([0,1], [0,1], 'r--')
    plt.plot(fnr, tpr)
    plt.title("EQUAL SAMPLES : AUC = "+str(calculateAUC(fnr, tpr)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    #--------------------------------------------------------------------------------------
    
    # UNEQUAL SAMPLES
    tpr = [1]
    fnr = [1]

    examples = [1,10, 20, 50, 100, 300, 500]

    for x in range(len(examples)):

        mu1 = [1, 0]
        sigma1 = [[1, .75], [0.75, 1]]
        mu2 = [0, 1]
        sigma2 = [[1, .75], [0.75, 1]]

        dataSet1 = dataSetLabel0(mu1, sigma1, examples[x])
        dataSet2 = dataSetLabel0(mu1, sigma1, 500)
        dataSet3 = dataSetLabel1(mu2, sigma2, 50)
        dataSet4 = dataSetLabel1(mu2, sigma2, 500)

        trainData0, trainLabel0, _, _ = dataSet1.generateData()
        _, _, testData0, testLabel0 = dataSet2.generateData()
        trainData1, trainLabel1, _, _ = dataSet3.generateData()
        _, _, testData1, testLabel1 = dataSet4.generateData()

        trainData = np.concatenate((trainData0, trainData1))
        testData = np.concatenate((testData0, testData1))
        trainLabel = np.concatenate((trainLabel0, trainLabel1))
        testLabel = np.concatenate((testLabel0, testLabel1))

        pred, posterior, err = myNB(trainData, trainLabel, testData, testLabel)

        tprate, fnrate = calculateTRPandFPR(pred, testLabel, posterior)
        tpr.append(tprate)
        fnr.append(fnrate)

    tpr.append(0)
    fnr.append(0)

    plt.plot([0,1], [0,1], 'r--')
    plt.plot(fnr, tpr)
    plt.title("UNEQUAL SAMPLES : AUC = "+str(calculateAUC(fnr, tpr)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


if __name__ == "__main__":
    main()


