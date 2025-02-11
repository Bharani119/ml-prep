class A:
    counter = 0
    def __new__(self):
        if self.counter == 1:
            raise Exception("Object cannot be created")
        self.counter += 1

a = A()

b = A() # this will result in an error