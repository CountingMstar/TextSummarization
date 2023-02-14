def yes(x):
    return 2*x



# x = 4
# a = lambda x: yes(x)+1
# print(a(5))

def no(x):
    a = lambda x: yes(x)+1
    print(a)
    return a 

no(10)
