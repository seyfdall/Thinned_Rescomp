import time

"""
Debugging Functions
"""
def time_comp(t0, message):
    t1 = time.time()
    print(f"{message}: {round(t1 - t0, 2)}")
    return t1