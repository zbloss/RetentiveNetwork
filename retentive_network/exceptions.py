class InvalidRetentionParametersException(Exception):
    """
    Raised in the event that parameters passed to the
    model are invalid according to the  architecture
    defined in the original paper.
    """

    def __init__(self, hidden_size: int, number_of_heads: int):
        self.message = f"hidden_size ({hidden_size}) must be divisible by number_of_heads ({number_of_heads})"
        super().__init__(self.message)
