import Skynet
import mnist_loader

if __name__ == '__main__':

    network = Skynet.Network([784, 30, 10])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    network.gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)
