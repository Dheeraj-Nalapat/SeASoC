"""
This is a sample module with a module docstring.
"""

def add_numbers(a, b):
    """
    Adds two numbers and returns the result.

    Parameters:
    - a (int): The first number.
    - b (int): The second number.

    Returns:
    int: The sum of the two numbers.
    """
    result = a + b
    return result

# Example usage
num1 = 10
num2 = 5
sum_result = add_numbers(num1, num2)
print(f"The sum of {num1} and {num2} is: {sum_result}")