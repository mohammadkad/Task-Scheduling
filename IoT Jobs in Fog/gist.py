import random
NUM_SENSORS = 21

'''
1404-05-11
random.randint(3, NUM_SENSORS//2), random.randint(3, 10) - This generates a random integer between 3 and 10, inclusive
Sample: 3, 4, 5, 6, 7, 8, 9, or 10.


random.sample(range(NUM_SENSORS), random.randint(3, NUM_SENSORS//2)) > random.sample(population, k)
a list of k distinct numbers from 0 to 20.
Sample: [20, 1, 12, 17, 8], [16, 6, 20, 15, 19], [4, 8, 3, 17, 11, 5, 20, 9, 0, 18]
'''
