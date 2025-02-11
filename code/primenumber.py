def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

l1 = [3, 7, 8]

for i in l1:
    print(is_prime(i))

# main statment to call the function
if __name__ == "__main__":
    print("This is a prime number program")
    n = int(input("Enter a number: "))
    if is_prime(n):
        print(f"{n} is a prime number")
    else:
        print(f"{n} is not a prime number")