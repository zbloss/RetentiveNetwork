import torch


class InvalidRetentionParametersException(Exception):
    """
    Raised in the event that parameters passed to the
    model are invalid according to the  architecture
    defined in the original paper.
    """

    def __init__(self, hidden_size: int, number_of_heads: int):
        self.message = f"hidden_size ({hidden_size}) must be divisible by number_of_heads ({number_of_heads})"
        super().__init__(self.message)


class InvalidTemperatureException(Exception):
    """
    Raised in the event that the temperature passed is not both greater than zero and less than or
    equal to 1.
    """

    def __init__(self, temperature: float):
        self.message = f"temperature ({temperature}) must be both greater than zero and less than or equal to 1"
        super().__init__(self.message)


class InvalidHiddenSizeException(Exception):
    """
    Raised in the event that the hidden size parameter passed is invalid
    """

    def __init__(self, hidden_size: int, model_required_hidden_size: int):
        self.message = f"hidden_size ({hidden_size}) must be equal to model.hidden_size ({hidden_size})"
        super().__init__(self.message)


class HalfPointPrecisionException(Exception):
    """
    Raised in the event that an operation attempts
    to use half point precision when Pytorch does not
    yet support operations on that type.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
    ):
        self.message = f"tensor (dtype: {tensor.dtype}) is not currently supported for this operation"
        super().__init__(self.message)


class ComplexTensorException(Exception):
    """
    Raised in the event that an operation attempts
    to use complex tensor dtypes when Pytorch does not
    yet support operations on that type.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
    ):
        self.message = f"tensor (dtype: {tensor.dtype}) is not currently supported for this operation"
        super().__init__(self.message)


class InvalidBatchSizeException(Exception):
    """
    Raised in the event that an operation attempts
    to tensors that have different batch_sizes. If
    this is raised, there is some improper tensor
    reshaping happening.

    """

    def __init__(
        self,
        tensora: torch.Tensor,
        tensorb: torch.Tensor,
    ):
        self.message = f"tensora (batch_size: {tensora.shape[0]}) is does not match tensorb (batch_size: {tensorb.shape[0]})"
        super().__init__(self.message)
