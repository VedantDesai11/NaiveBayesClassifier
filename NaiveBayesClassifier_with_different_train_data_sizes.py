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
        for key, value in meanSDCount_trainData.items():

            #probabiltyOfLabel
            probabilty[key] = value[0][2]/numberOfTestData

            for i in range(len(data)):
                probabilty[key] *= calculateProbabilty(data[i], value[i][0], value[i][1])

        highest = 0
        for key, value in probabilty.items():

            if value > highest:
                highest = value
                label = key

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


def main():

    listOfAccuracy = []

    examples = [10, 20, 50, 100, 300, 500]

    for x in range(len(examples)):

        mu1 = [1, 0]
        sigma1 = [[1, .75], [0.75, 1]]
        mu2 = [0, 1]
        sigma2 = [[1, .75], [0.75, 1]]
        dataSet1 = dataSetLabel0(mu1, sigma1, examples[x])
        dataSet2 = dataSetLabel0(mu1, sigma1, 500)
        dataSet3 = dataSetLabel1(mu2, sigma2, examples[x])
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
        listOfAccuracy.append(1-err)


    plt.title("Accuracy vs Number of Training Data")
    plt.plot(listOfAccuracy, examples)
    plt.show()


if __name__ == "__main__":
    main()


