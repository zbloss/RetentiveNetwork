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
