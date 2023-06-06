from models.loss.LossClass import LossClass

class TorchLoss(LossClass):
    """
    A class representing a loss function and optimizer for PyTorch model compilation.

    The TorchLoss class extends the LossClass and provides an implementation for specifying
    a loss function and optimizer specifically for PyTorch models.

    """

    def __init__(self, loss_function, optimizer):
        """
        Initializes a TorchLoss object with the specified loss function and optimizer.

        :param loss_function: The loss function to be used for PyTorch model training.
        :param optimizer: The optimizer algorithm to be used for PyTorch model optimization.
        """
        self.loss_function = loss_function
        self.optimizer = optimizer

    def compile_model(self, model):
        """
        Compiles a PyTorch model with the specified loss function and optimizer.

        :param model: The PyTorch model to be compiled.
        :return: None
        """
        model.loss_function = self.loss_function
        model.optimizer = self.optimizer
