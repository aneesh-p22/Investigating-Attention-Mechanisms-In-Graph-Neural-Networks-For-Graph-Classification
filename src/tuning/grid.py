from itertools import product


def generate_grid(search_space):
    names = list(search_space.keys())

    value_lists = [
        search_space[name]
        for name in names
    ]

    configurations = []

    for combination in product(*value_lists):
        configuration = dict(
            zip(names, combination)
        )

        configurations.append(configuration)

    return configurations