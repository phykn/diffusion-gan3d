from dataclasses import MISSING, field


def number(default=MISSING, *, minimum=0.0, maximum=None, exclusive=False):
    return field(
        default=default,
        metadata={"minimum": minimum, "maximum": maximum, "exclusive": exclusive},
    )


def integer(default=MISSING, *, minimum=1):
    return field(default=default, metadata={"minimum": minimum})


def probability(default):
    return number(default, maximum=1.0)


def fraction(default):
    return number(default, maximum=1.0, exclusive=True)


def check_range(setting, value, path):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    minimum = setting.metadata.get("minimum")
    maximum = setting.metadata.get("maximum")
    exclusive = setting.metadata.get("exclusive", False)
    if minimum is not None and value < minimum:
        if type(value) is int:
            bound = {0: "a non-negative integer", 1: "a positive integer"}.get(
                minimum, f"an integer of at least {minimum}"
            )
            raise ValueError(f"{path} must be {bound}.")
        raise ValueError(f"{path} must be at least {minimum}.")
    if maximum is not None and (value >= maximum if exclusive else value > maximum):
        end = ")" if exclusive else "]"
        raise ValueError(f"{path} must be in [{minimum}, {maximum}{end}.")
