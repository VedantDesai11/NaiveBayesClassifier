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


def calculatePrecisionAndRecall(pred, testLabel):

    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    for x in range(len(pred)):
        if pred[x] == 1 and testLabel[x] == 1:
            truePositive += 1
        if pred[x] == 0 and testLabel[x] == 0:
            trueNegative += 1
        if pred[x] == 1 and testLabel[x] == 0:
            falsePositive += 1
        if pred[x] == 0 and testLabel[x] == 1:
            falseNegative += 1

    matrix = [[truePositive,falsePositive], [falseNegative,trueNegative]]

    precision = truePositive/(truePositive+falsePositive)
    recall = truePositive/(falseNegative+truePositive)

    return precision, recall, np.array(matrix)


def plotData(trainData0, trainData1, testData0, testData1, testData, pred):
    plotLabel0 = []
    plotLabel1 = []

    plt.scatter(trainData0[:, 0], trainData0[:, 1])
    plt.scatter(trainData1[:, 0], trainData1[:, 1])
    plt.title("training data label 0 and label 1")
    plt.show()

    for i in range(len(list(testData))):
        if pred[i] == 0:
            plotLabel0.append(testData[i])
        else:
            plotLabel1.append(testData[i])

    plotLabel0 = np.array(plotLabel0)
    plotLabel1 = np.array(plotLabel1)

    fig, axs = plt.subplots(1, 2)

    plt.title("Predicted Classes vs Actual Classes")

    axs[0].scatter(plotLabel0[:, 0], plotLabel0[:, 1])
    axs[0].scatter(plotLabel1[:, 0], plotLabel1[:, 1])

    axs[1].scatter(testData0[:, 0], testData0[:, 1])
    axs[1].scatter(testData1[:, 0], testData1[:, 1])

    plt.show()


def main():

    listOfAccuracy = []
    listOfPrecision = []
    listOfRecall = []
    listOfConfusionMatrix = []

    for x in range(10):

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
        listOfAccuracy.append(1-err)

        precision, recall, confusionMatrix = calculatePrecisionAndRecall(pred, testLabel)
        listOfPrecision.append(precision)
        listOfRecall.append(recall)
        listOfConfusionMatrix.append(confusionMatrix)

        if x == 0:

            print("Accuracy for test: " + str(1-err))
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("Confusion Matrix")
            print(confusionMatrix)

            plotData(trainData0, trainData1, testData0, testData1, testData, pred)


    print("1000 Training Data, 1000 Testing Data")
    print("Prediction Accuracy")
    print(listOfAccuracy)
    print("average accuracy: "+str(sum(listOfAccuracy) / len(listOfAccuracy)))
    print("-----------------------------------------------------------")
    print("Precision")
    print(listOfPrecision)
    print("average precision: "+str(sum(listOfPrecision) / len(listOfPrecision)))
    print("-----------------------------------------------------------")
    print("Recall")
    print(listOfRecall)
    print("average recall for 0: "+str(sum(listOfRecall) / len(listOfRecall)))
    print("-----------------------------------------------------------")
    print("List of Confusion matrices")
    for matrix in listOfConfusionMatrix:
        print(matrix)
        print("---------")


if __name__ == "__main__":
    main()


