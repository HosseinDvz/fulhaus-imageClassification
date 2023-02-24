# fulhaus-imageClassification

To make the job easier and leverage the torchvision datasets.ImageFolder, I created a test and train folder.
I chose 20 images from each class for test dataset and put them in test folder.
in this way while torchvision reads the data from folders, automatically creates labels for them.

## About the transformations:
To have a more robust model, I decided to not to apply to original images to the model which seemed pretty easy for models to predict.
the following transformation will be applied while reading the data from folders:
- rotating by 10 degree
- flipping by %50 chance
- resizing to have smaller pictures
- center cropping (keeping the center of the images)
- tranform it to a tensor 
- and finally normilized them. the mean and standard deviation for normalization come from best practices.

## About the CNN model:
Probably for any image classification, the first model that comes to mind is CNN model.
I have another repository in which I have explained how CNN models work.
I had already worked with this model and it did not take too much time for me to come up with a model that works perfect 
with this dataset.

Since the images were not chalenging, I chose 6 filters (3x3) for the first convolution layer with stride of one, 16 filters for second layer, 
followed by  2x2 maxpolling after each conv layers and three linear layers. 
For hidden layer activation functions I chose Relu, and for the output layer, I chose log_softmax since it is a multi calss classification and 
compare to softmax, it more penalize the wrong classifications. After having a look at training data, I decided to do this since images were very clear.

I played with batch size and number of epochs and finally came up with batches of 10 and epochs of 15 and here is the result of last epoch:

```
epoch:  4  batch:    8 [    80/240]  loss: 0.24419129  accuracy:  98.750%
epoch:  4  batch:   10 [   100/240]  loss: 0.10034236  accuracy:  98.000%
epoch:  4  batch:   12 [   120/240]  loss: 0.31954050  accuracy:  95.833%
epoch:  4  batch:   14 [   140/240]  loss: 0.01942632  accuracy:  95.714%
.
.
.
epoch: 14  batch:   12 [   120/240]  loss: 0.00119323  accuracy:  99.167%
epoch: 14  batch:   14 [   140/240]  loss: 0.01600687  accuracy:  99.286%
epoch: 14  batch:   16 [   160/240]  loss: 0.00025624  accuracy:  99.375%
epoch: 14  batch:   18 [   180/240]  loss: 0.02391354  accuracy:  99.444%
epoch: 14  batch:   20 [   200/240]  loss: 0.13720807  accuracy:  99.000%
epoch: 14  batch:   22 [   220/240]  loss: 0.00039691  accuracy:  98.636%
epoch: 14  batch:   24 [   240/240]  loss: 0.00029440  accuracy:  98.750%
epoch: 15  batch:    2 [    20/240]  loss: 0.00102266  accuracy: 100.000%
epoch: 15  batch:    4 [    40/240]  loss: 0.05912386  accuracy: 100.000%
epoch: 15  batch:    6 [    60/240]  loss: 0.00191698  accuracy: 100.000%
epoch: 15  batch:    8 [    80/240]  loss: 0.00337010  accuracy: 100.000%
epoch: 15  batch:   10 [   100/240]  loss: 0.00067392  accuracy: 100.000%
epoch: 15  batch:   12 [   120/240]  loss: 0.00027475  accuracy: 100.000%
epoch: 15  batch:   14 [   140/240]  loss: 0.00085458  accuracy: 100.000%
epoch: 15  batch:   16 [   160/240]  loss: 0.00165798  accuracy: 100.000%
epoch: 15  batch:   18 [   180/240]  loss: 0.00167200  accuracy: 100.000%
epoch: 15  batch:   20 [   200/240]  loss: 0.00151812  accuracy: 100.000%
epoch: 15  batch:   22 [   220/240]  loss: 0.00061435  accuracy: 100.000%
epoch: 15  batch:   24 [   240/240]  loss: 0.00892481  accuracy: 100.000%
```
