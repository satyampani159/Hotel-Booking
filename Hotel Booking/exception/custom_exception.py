class CustomException(Exception):
    def __init__(self, message: str, errors: Exception = None):
        super().__init__(message)
        self.errors = errors
