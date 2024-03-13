import time


def repeat_test(min_duration, min_iterations):
    end = time.time() + min_duration
    for _ in range(min_iterations):
        yield
    while time.time() < end:
        yield


def median_attr(objects, attr):
    values = []
    for obj in objects:
        values.append(getattr(obj, attr))
    values.sort()
    n = len(values)
    return (values[n // 2] + values[(n - 1) // 2]) / 2
