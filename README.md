# CNN-from-scratch-using-Numpy
Implementation of Convolutional Neural Network from scratch using Numpy - An internship project under Dr. Girish Varma at IIIT Hyderabad.


The Convolutional Neural Network was implemented on the mnist dataset for handwritten digits recognition.

The neural network consists of three hidden layers. The input is passed through a convolutional network and pooled to give us the first hidden layer. The output of the previous step is then passed through a convolutional network and pooled again which is then flattened to give us the second hidden layer. This is now passed through a fully connected linear neural network which makes the 3rd hidden layer, which is finally passed through another fully connected linear neural network to give us the output.

The accuracy which will be attained by running this code will depend on the number of epochs. The best accuracy recorded was around 92% on the training set and around 91% in the test set in the same run.

The present written code will terminate once the accuracy crosses 85% and will show the accuracy on the test set (the limit of 85% can be altered).

The code was run on Google Colab. Link: https://colab.research.google.com/drive/1Sx-e10vP6oaSQNBjakBXW1BkQe4bdmWb?usp=sharing

Details of the internship:

Guide: Dr. Girish Varma (Homepage: https://girishvarma.in)

At: IIIT Hyderabad

Topic: Machine Learning and its theory

Participants: Pulkit Thakar (GitHub: PulkitThakar),
              Srinidhi Kulkarni (GitHub: SrinidhiKulkarni28)
