import concurrent.futures
import time


def square_number(n):
    print(f"Calculating square of {n}")
    time.sleep(1)  # Simulating a time-consuming task
    return n * n


# Create a ThreadPoolExecutor with 2 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    # Submit tasks to the executor
    future1 = executor.submit(square_number, 5)
    future2 = executor.submit(square_number, 8)

    # Retrieve results from the futures
    result1 = future1.result()
    result2 = future2.result()

    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}")
