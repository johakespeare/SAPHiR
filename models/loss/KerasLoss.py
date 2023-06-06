from models.loss.LossClass import LossClass

class KerasLoss(LossClass):
    """
    A class representing a loss function and optimizer for Keras model compilation.

    The KerasLoss class extends the LossClass and provides an implementation for specifying
    a loss function and optimizer specifically for Keras models.

    """

    def __init__(self, loss_function, optimizer):
        """
        Initializes a KerasLoss object with the specified loss function and optimizer.

        :param loss_function: The loss function to be used for Keras model training.
        :param optimizer: The optimizer algorithm to be used for Keras model optimization.
        """
        self.loss_function = loss_function
        self.optimizer = optimizer

    def compile_model(self, model):
        """
        Compiles a Keras model with the specified loss function and optimizer.

        :param model: The Keras model to be compiled.
        :return: None
        """
        model.compile(loss=self.loss_function, optimizer=self.optimizer)
