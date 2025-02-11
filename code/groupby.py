from itertools import groupby

# Sample list of tuples
data = [
    ('apple', 1),
    ('banana', 2),
    ('apple', 3),
    ('banana', 1),
    ('orange', 2),
    ('apple', 2)
]

# Sort the data by the first element of the tuples (the key)
data.sort(key=lambda x: x[0])

# Group by the first element of the tuples
grouped_data = {key: list(group) for key, group in groupby(data, key=lambda x: x[0])}

# Print the grouped data
print(grouped_data)
