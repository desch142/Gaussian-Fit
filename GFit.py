import numpy as np


class GFit():
    def __init__(self, number, string):
        self.n = number
        self.str = string

    def test(self):
        print(self.str)

test = GFit(6, 'hey')
np.arange(0,10)
print(test.str)
test.test()