import random
import supervised

def data_shuffle(dataSet):
    with open(dataSet,"r") as f1, open("Shuffled_data.csv","w") as f2:
        lines = f1.readlines()
        cpy = str(lines[0])
        random.shuffle(lines)
        f2.write(cpy)
        for line in lines:
            if line != cpy:
                f2.write(line)

def classifier(dataSet, shuffling = 0, yCol = -1):
    if shuffling == 1:
        data_shuffle(dataSet)
        classifier("Shuffled_data.csv",0)
    else:
        print("Computing...")
        algorithmName, highAccuracy, savedModel = supervised.classifier(dataSet,yCol)
        return algorithmName, highAccuracy, savedModel

if __name__ =='__main__':
    classifier(dataSet,shuffing,yCol)
    #cluster(dataSet,shuffling)
