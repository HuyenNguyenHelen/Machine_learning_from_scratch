class Fibonnaci():
    def fibo(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        else:
            return self.fibo(n-1) + self.fibo(n-2)

func = Fibonnaci()
print(func.fibo(10))
    
        