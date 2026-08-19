def diff(f,x,eps=1e-6):
    return (f(x+eps)-f(x))/eps
def Newton_iter(f,x):
    return x-diff(f,x)/diff(lambda x:diff(f,x),x)
def optimize(f,x_start,min_dif=1e-3):
    x0=x_start
    x1=Newton_iter(f,x0)
    i=1
    while abs(x1-x0)>min_dif:
        print(i,x1)
        x0=x1
        x1=Newton_iter(f,x0)
        i+=1
    print(i,x1)

