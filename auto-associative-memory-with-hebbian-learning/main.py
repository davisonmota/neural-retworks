import math

W = []

class HebbianMemory:
    def __init__(self, input_size):
        self.W = [[0 for _ in range(input_size)] for _ in range(input_size)]
        self.input_size = input_size

    def train_package(self, patterns):
        for pattern in patterns:
            self.train(pattern)

    def train(self, pattern):
        for i in range(self.input_size):
            for j in range(self.input_size):
                self.W[i][j] += pattern[i] * pattern[j] # XX^T

    def recall(self, pattern):
        memory_pattern = [0] * self.input_size
        for i in range(self.input_size):
            for j in range(self.input_size):
                memory_pattern[i] += self.W[i][j] * pattern[j]

        return memory_pattern

    def vector_product(self, x, y):
        product = 0
        for i in range(len(x)):
            product += x[i] * y[i]
        return product

    def cosine_similarity(self, returned_pattern, pattern):
        product = self.vector_product(returned_pattern, pattern) # produto matricial entre o vetor returned_pattern transposta por pattern
        norm_returned_pattern =  math.sqrt(self.vector_product(returned_pattern, returned_pattern))
        norm_pattern =   math.sqrt(self.vector_product(pattern, pattern))
        return product / (norm_returned_pattern * norm_pattern)



memory=HebbianMemory(4)
memory.train_package([
    [1,1,1,-1],
    [-1,-1,1,-1],
    [-1,-1,-1,-1],
    [1,-1,-1,1],
    [-1,1,-1,-1],
    [1,-1,1,1],
    [1,1,1,1],
    [-1,-1,-1,1],
    [-1,1,1,-1],
    [1,1,-1,1]
])

print("Training patterns:")
for row in memory.W:
    print(row)


print("Recall pattern:")
returned_pattern = memory.recall([1,1,1,-1])
print(returned_pattern)

print("----------------------")
print(memory.cosine_similarity(returned_pattern, [-1,1,1,-1]))