def is_palindrome(s):
    s = s.lower().replace(" ", "")  # Convert to lowercase and remove spaces
    print(s[::-1])
    return s == s[::-1]  # Check if the string is equal to its reverse


# print(is_palindrome("bharani"))

# li = "bharani"
# sorted(li, reverse=True)
# import numpy as np
# import pandas as pd

# li.split("")
# pd.Series(np.array([char for char in li])).value_counts()


def log_function_name(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


@log_function_name
def greet(name):
    return f"Hello, {name}!"


print(greet("Alice"))


import time

start_time = time.time()
end_time = start_time + 5

for i in range(1000000):  # Just an example range
    if time.time() > end_time:
        break
    # Your code here
    print("Inside the loop")

# print("Loop finished after 5 seconds")
