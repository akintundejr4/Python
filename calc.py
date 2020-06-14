import os
import re
import sys

def add(x, y): 
    """Add two numbers together"""
    return x + y

def subtract(x, y): 
    """Subtract two numbers"""
    return x - y 

def multiply(x, y): 
    """Multiply two numbers"""
    return x * y

def divide(x, y): 
    """Divide two numbers"""
    if y == 0: 
        raise ValueError("Error - Division by zero!")
    return x / y