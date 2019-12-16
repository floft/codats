"""
Recursively print dictionary (e.g. use with collections.defaultdict(dict))
"""


def _quote_if_string(value):
    if isinstance(value, str):
        return "\"" + value + "\""
    else:
        return str(value)


def _print_dictionary(d, name, prepend=""):
    """ Recursively print dictionary """
    print(prepend + _quote_if_string(name) + ": {")

    for k in d.keys():
        if isinstance(d[k], dict):
            _print_dictionary(d[k], k, prepend=prepend+"    ")
        else:
            print(prepend + "    " + _quote_if_string(k) + ": " + _quote_if_string(d[k]) + ",")

    print(prepend + "},")


def print_dictionary(d, name):
    """ Recursively print dictionary """
    print(name, "= {")

    for k in d.keys():
        if isinstance(d[k], dict):
            _print_dictionary(d[k], k, prepend="    ")
        else:
            print("    " + _quote_if_string(k) + ": " + _quote_if_string(d[k]) + ",")

    print("}")
