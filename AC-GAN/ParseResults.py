import numpy as np

def ParseAndReport(filename):

    results = open(filename, "r")

    resultsList = []
    for line in results:
        splitLine = line.split(": ")
        splitLine = splitLine[-1].strip("\n")
        resultsList.append(float(splitLine))

    print ("Mean value: " + str(np.mean(resultsList)) + " with stddev: " + str(np.std(resultsList)))

print ("***RESULTS FOR 10 SAMPLES PER CLASS***")
ParseAndReport("ResultsDiscriminator10")
print ("***RESULTS FOR 100 SAMPLES PER CLASS***")
ParseAndReport("ResultsDiscriminator100")
print ("***RESULTS FOR ENTIRE MNIST***")
ParseAndReport("ResultsDiscriminatorEntire")
