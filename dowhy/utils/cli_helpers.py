def query_yes_no(question, default=True):
    """Ask a yes/no question via standard input and return the answer.

    Source: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input

    If invalid input is given, the user will be asked until they actually give valid input.

    Side Effects: Blocks program execution until valid input(y/n) is given.

    :param question(str):  A question that is presented to the user.
    :param default(bool|None): The default value when enter is pressed with no value.
        When None, there is no default value and the query will loop.
    :returns: A bool indicating whether user has entered yes or no.
    """
    yes_list = ["yes", "y"]
    no_list = ["no", "n"]
    default_dict = {None: "[y/n]", True: "[Y/n]", False: "[y/N]"}  # default => prompt default string
    default_str = default_dict[default]
    prompt_str = "%s %s " % (question, default_str)
    while True:
        choice = input(prompt_str).lower()
        if not choice and default is not None:
            return default
        if choice in yes_list:
            return True
        if choice in no_list:
            return False
        notification_str = "Please respond with 'y' or 'n'"
        print(notification_str)
