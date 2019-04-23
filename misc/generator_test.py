"""
This illustrates how we should write iterators in scripts for parallel
simulations (using imap_unordered) without unnecessary calculations.
"""


def test_gen():
    for i in range(4):
        print(f'i={i:d}')
        for j in range(i):
            print(f'j={j:d}')
            yield i+j


for k in test_gen():
    print(f'i+j={k:d}')


