import os


def get_env_variable(var_name):
    """
    Get the environment variable value. Raise an error if not found.

    :param var_name: str, environment variable name.
    :return: str, the value of the environment variable.
    """
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(
            f"Environment variable '{var_name}' not found. Please set it in your .env file or system environment."
        )
    return value
