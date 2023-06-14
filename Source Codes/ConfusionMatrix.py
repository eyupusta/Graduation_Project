import math

class confusionMatrix:
    def __init__(self, matrix = [[]]):
        self.tp = matrix[0][0] #True Positive
        self.fn = matrix[0][1] #False Negative
        self.fp = matrix[1][0] #False Positive
        self.tn = matrix[1][1] #True Negative
    def accuracy(temp):
        sumVrb = temp.tn + temp.tp
        total = sumVrb + temp.fp + temp.fn
        return sumVrb / total
    def recall(temp): #True Positive Rate
        sumVrb = temp.tp + temp.fn
        return temp.tp / sumVrb
    def specificity(temp): #True Negative Rate
        sumVrb = temp.tn + temp.fp
        return temp.tn / sumVrb
    def precision(temp): #Positive Predictive Value
        sumVrb = temp.tp + temp.fp
        return temp.tp / sumVrb
    def npv(temp): #Negative Predictive Value
        sumVrb = temp.tn + temp.fn
        return temp.tn / sumVrb
    def missRate(temp): #False Negative Rate 
        sumVrb = temp.fn + temp.tp
        return temp.fn / sumVrb
    def fallOut(temp): #False Positive Rate
        sumVrb = temp.fp + temp.tn
        return temp.fp / sumVrb
    def fdr(temp): #False Discovery Rate
        sumVrb = temp.fp + temp.tp
        return temp.fp / sumVrb
    def forFunc(temp): #False Omission Rate
        sumVrb = temp.fn + temp.tn
        return temp.fn / sumVrb
    def threatScore(temp): #Critical Success Index
        sumVrb = temp.tp + temp.fn + temp.fp
        return temp.tp / sumVrb
    def prevalenceThreshold(temp):
        value1 = confusionMatrix.specificity(temp) * (-1)
        value1 += 1
        value1 *= confusionMatrix.recall(temp)
        value1 = math.sqrt(value1)
        value1 = value1 + confusionMatrix.specificity(temp) - 1
        value2 = confusionMatrix.recall(temp) + confusionMatrix.specificity(temp) - 1
        return value1 / value2
    def fMeasure(temp):
        value1 = confusionMatrix.precision(temp) * confusionMatrix.recall(temp)
        value2 = confusionMatrix.precision(temp) + confusionMatrix.recall(temp)
        return value1 / value2
    def mcc(temp): #Matthews Correlation Coefficient
        value1 = temp.tp * temp.tn - temp.fp * temp.fn
        value2 = (temp.tp + temp.fp) * (temp.tp + temp.fn) * (temp.tn + temp.fp) * (temp.tn + temp.fn)
        value2 = math.sqrt(value2)
        return value1 / value2
    def fowlkesMallows(temp): #Fowlkes–Mallows Index
        value1 = confusionMatrix.precision(temp) * confusionMatrix.recall(temp)
        return math.sqrt(value1)
    def informedness(temp): #Bookmaker Informedness
        return confusionMatrix.recall(temp) + confusionMatrix.specificity(temp) - 1
    def markedness(temp): #deltaP(Δp)
        return confusionMatrix.precision(temp) + confusionMatrix.npv(temp) - 1
