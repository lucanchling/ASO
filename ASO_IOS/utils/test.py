from functools import update_wrapper


def count(func):
    count = 0 
    def f(*args):
        print('arg',args[1])
        f.count+=1
        func(*args)
        count = f.count
    f.count = 0
    return f

class Test:
    def __init__(self) -> None:
        self.var = 'gdgdeg'

    @count
    def f(self,f,s,dsgh,g):
        # print(f,s)
        print(self.var)


a = Test()
for i in range(5):
    a.f(i,i,i,i)

print(a.f.count)

