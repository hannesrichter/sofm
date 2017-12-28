'''This file makes the data from the AwA2 dataset available to the animals sofm experiment
All data was taken from https://cvml.ist.ac.at/AwA2/'''

features = [[0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,0],
[1,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0],
[1,1,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1,0,0,0],
[0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,0,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0],
[1,1,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,1,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,1],
[0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1],
[1,1,0,1,1,0,0,0,1,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,1],
[1,0,0,1,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,1],
[0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,0],
[1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,0,1],
[1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0],
[1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,1,1,0,1,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0,1,1,0],
[1,1,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1,1,1,1,0],
[0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0],
[1,0,0,1,0,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,0,1,1,0],
[0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,0,1,1,0,0],
[1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0],
[1,0,1,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,0,0],
[0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,0,0],
[1,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0],
[1,1,0,1,1,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,1],
[0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,1,1,0],
[1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,0,1],
[1,1,0,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,0,1],
[1,0,0,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,0,1,0,1,1,1,1,0,1,1],
[1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,1],
[0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,1,1,0],
[0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0],
[1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,1,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,1,1],
[1,0,0,1,1,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,1,1,0,1,0],
[0,0,0,1,0,1,0,1,1,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,0],
[1,1,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,0],
[1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,0,1],
[1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,1,0,1,1,0,0,1,0,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,1,0,1,1,0],
[1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,1,0,1,0,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0],
[1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,1,1,0],
[1,0,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0],
[1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0],
[1,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,1,0,0,1,0,1,0,0,1,1,1,1,1,1],
[0,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0],
[0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,1,0,0],
[1,1,0,1,1,0,0,0,1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1],
[0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,1,1,0,1,0],
[0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,0,1,1,1,0,0,0,1,0,0,1,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,1,0,1,1],
[0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,0,0],
[0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1],
[0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,1,0],
[1,1,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0],
[1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,1],
[0,1,1,0,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,0,1]]


animals = ['antelope','grizzly bear','killer whale','beaver','dalmatian','persian cat','horse','german shepherd','blue whale','siamese cat','skunk','mole','tiger','hippopotamus','leopard','moose','spider monkey','humpback whale','elephant','gorilla','ox','fox','sheep','seal','chimpanzee','hamster','squirrel','rhinoceros','rabbit','bat','giraffe','wolf','chihuahua','rat','weasel','otter','buffalo','zebra','giant panda','deer','bobcat','pig','lion','mouse','polar bear','collie','walrus','raccoon','cow','dolphin']