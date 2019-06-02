# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        # set default values and data structures
        weightsLearned = util.Counter()
        weights = util.Counter()
        score = util.Counter()
        updatedFeatures = util.Counter()
        tdLength = len(trainingData)
        vdLength = len(validationLabels)
        trainingDataKeys = trainingData[0].keys()
        maxScore = float("-inf")
        addWeight = 0
        addOne = 1
        tau = 0


        # loop through held-out validation set for each C
        for c in Cgrid:
            # pass through the data self.max_iterations
            for iterations in range(self.max_iterations):
                # loop through the training data
                for i in range(tdLength):
                    # loop through each legal label
                    # y' = arg max score(F, y'')
                    for label in self.legalLabels:
                        score[label] = trainingData[i].__mul__(self.weights[label])

                    maxScore = score.argMax()
                    tau = 0
                    # if y' == y do nothing otherwise update weight vectors
                    if (trainingLabels[i] != maxScore):
                        C = ((self.weights[maxScore].__sub__(self.weights[trainingLabels[i]])).__mul__(trainingData[i]) + 1.0 ) / \
                            (2.0 * (trainingData[i].__mul__(trainingData[i])))

                        # cap the maximum possible values of tau by a positive constant c
                        tau = min(c, C)
                        updatedFeatures.clear()

                        # tau * f
                        for label in trainingDataKeys:
                            feature = trainingData[i][label]
                            updatedFeatures[label] = tau * feature

                        # update the weight vectors of labels with variable
                        self.weights[trainingLabels[i]].__radd__(updatedFeatures)
                        self.weights[maxScore].__sub__(updatedFeatures)

            weights[c] = self.weights

            for i in range(vdLength):
                for label in validationLabels:
                    score[label] = validationData[i].__mul__(self.weights[label])

                maxScore = score.argMax()

                if validationLabels[i] == maxScore:
                    addWeight = addWeight + addOne

            weightsLearned[c] = addWeight

        maxScore = weightsLearned.argMax()

        # storing the weights learning using the best value of C
        self.weights = weights[maxScore]

        # util.raiseNotDefined()

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


