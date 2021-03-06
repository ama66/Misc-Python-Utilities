## From Corey Schafer tutorials 
from pympler import summary, muppy
import psutil
import resource
import os
import sys
import random
import time


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def memory_usage_resource():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem


def square_numbers(nums):
    for i in nums:
        yield (i*i)

my_nums = square_numbers([1,2,3,4,5])



print("First value")


print(next(my_nums))

print("Second value")

print(next(my_nums))

print("looping over the rest")

for num in my_nums:
    print(num)


print('Another way to make generators like list comprehension')
my_nums = (x*x for x in [1,2,3,4,5])
print(next(my_nums))
# print(list(my_nums)) # [1, 4, 9, 16, 25]
# print(my_nums)

########## Demonstrate memory and run-time advantage
names = ['John', 'Corey', 'Adam', 'Steve', 'Rick', 'Thomas']
majors = ['Math', 'Engineering', 'CompSci', 'Arts', 'Business']

print ('Memory (Before): {}Mb'.format(memory_usage_psutil()))

def people_list(num_people):
    result = []
    for i in range(num_people):
        person = {
                    'id': i,
                    'name': random.choice(names),
                    'major': random.choice(majors)
                }
        result.append(person)
    return result

def people_generator(num_people):
    for i in range(num_people):
        person = {
                    'id': i,
                    'name': random.choice(names),
                    'major': random.choice(majors)
                }
        yield person

t1 = time.clock()
people = people_list(1000000)
t2 = time.clock()


print('Memory (After) : {}Mb'.format(memory_usage_psutil()))
print('List Took {} Seconds'.format(t2-t1))



t1 = time.clock()
people = people_generator(1000000)
t2 = time.clock()

print('Memory (After) : {}Mb'.format(memory_usage_psutil()))
print('Generator Took {} Seconds'.format(t2-t1))

## First Class Functions 
def square(x):
	return x*x 

def html_tag(tag):

    def wrap_text(msg):
        print('<{0}>{1}</{0}>'.format(tag, msg))

    return wrap_text

f=square
print(f)
print(f(5))

############ 
print_h1 = html_tag('h1') 

## Returns a function with h1  as emebedded parameter / closure 

print_h1('Test Headline!')

print_h1('Another Headline!')

print_p = html_tag('p')

print_p('Test Paragraph!')

def my_map(f,lis):
	g=[]
	for i in lis:
		g.append(f(i))
	return g
print("my_map higher order function ")
print(my_map(f,[1,2,3,4,5]))



##Snippets 

# Number of days per month. First value placeholder for indexing purposes.
# Number of days per month. First value placeholder for indexing purposes.
month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap(year):
    """Return True for leap years, False for non-leap years."""

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year, month):
    """Return number of days in that month in that year."""

    # year 2017
    # month 2
    if not 1 <= month <= 12:
        return 'Invalid Month'

    if month == 2 and is_leap(year):
        return 29

    return month_days[month]

print(days_in_month(2017, 2))


def student_info(*args,**kwargs):
	print('positional arguments are a tuple')
	print(args)
	print('keyword arguments are a dictionary')

	print(kwargs)

student_info('Math','History',2,3,name='Ali', Age=22)

dic=['Math','History',2,3]
info={'name': 'Ali', 'Age': 22}
student_info(*dic,**info)  ## * is used to unpack stuff into args and kwargs




########### Closures ############

import logging

logging.basicConfig(filename='example.log', level=logging.INFO)


def logger(func):
    def log_func(*args):
        logging.info(
            'Running "{}" with arguments {}'.format(func.__name__, args))
        print(func(*args))
    return log_func


def add(x, y):
    return x+y


def sub(x, y):
    return x-y

add_logger = logger(add)
sub_logger = logger(sub)

add_logger(3, 3)
add_logger(4, 5)

sub_logger(10, 5)
sub_logger(20, 10)


######## Decorators - Dynamically Alter The Functionality Of Your Functions 

def outer_funct(message):
	mesg=message
	def inner_func():
		print(mesg)
	return inner_func 

# my_func=outer_funct()

# my_func()

hi_func=outer_funct('Hi')
hi_func()
hello_func=outer_funct('Hello')
hello_func()

#####################################

print("\n")
print("*"*40)
def decorator_function(original_function):

	def wrapper_function(*args,**kwargs):
		## Added functionality
		print("Wrapper Added this before running {}".format(original_function.__name__))
		return original_function(*args,**kwargs)
	return wrapper_function



@decorator_function
def display():
	print("Display function ran! ")


@decorator_function
def display_info(name,age):
	print("Display info ran with arguments({},{})".format(name,age))

display_info("Ammar", 41)




#decorated_display=decorator_function(display)

#decorated_display()

display()
print("*"*40)


################ Decorator Class

# class decorator_class(object):

#     def __init__(self, original_function):
#         self.original_function = original_function

#     def __call__(self, *args, **kwargs):
#         print('call method before {}'.format(self.original_function.__name__))
#         self.original_function(*args, **kwargs)


# @decorator_class
# def display_me(name,age):
# 	print("Display me ran with arguments({},{})".format(name,age))

# display_me("Ammar", 41)


############# Practical Examples ############ 
# Practical Examples

def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

import time


##################

from functools import wraps


def my_logger(orig_func):
    import logging
    logging.basicConfig(filename='{}.log'.format(orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
        return orig_func(*args, **kwargs)

    return wrapper


def my_timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result

    return wrapper

import time


@my_logger
@my_timer
def display_info(name, age):
    time.sleep(1)
    print('display_info ran with arguments ({}, {})'.format(name, age))

display_info('Tom', 22)






