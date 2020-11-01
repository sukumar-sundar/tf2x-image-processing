from TF2XImages.ConvolutionalNeuralNet.basic_cnn.BasicCNN import BasicCNN

#Basic model
def basic_cnn_modelling_caller(info):
    try:

        #/Users/vvsukumarsundararajan/.keras/datasets/cifar-10-batches-py.tar.gz
        print(f'Hi, {info}')
        bcn = BasicCNN()

        #Collect the images.
        bcn.collectCIFAR10Data()

        #Validate the images
        bcn.verifyData()

        #Create Model
        bcn.create_convolutional_model()

    except Exception as exc:
        print("Caught in the unexpected exception {0}".format(exc))


# Advanced Model




# Start from here.
if __name__ == '__main__':
    basic_cnn_modelling_caller('Calling the image classification function...')




