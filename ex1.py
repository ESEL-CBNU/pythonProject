# python grammer excercise
data = [10, 25, 30, 45, 50]
for i in data:
    if i > 0:
        print("임계값 초과:", i)
    else:
        print("정상:")
        
# python class excercise
class Car:
    # property initialization
    def __init__(self, model, year):
        self.model = model
        self.year = year
    # method to display car information
    def display_info(self):
        print(f"Model: {self.model}, Year: {self.year}")
# Instance creation
car1 = Car("Tesla", 2020)
car2 = Car("BMW", 2019)
# Displaying car information
car1.display_info()
car2.display_info()

# class excercise with inheritance
class ElectricCar(Car):
    def __init__(self, model, year, battery_capacity):
        super().__init__(model, year)
        self.battery_capacity = battery_capacity
    def display_info(self):
        super().display_info()
        print(f"Battery Capacity: {self.battery_capacity} kWh")
# Instance creation of ElectricCar
e_car = ElectricCar("Nissan Leaf", 2021, 40)
e_car.display_info()

# python exception handling excercise
def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except TypeError:
        return "Error: Invalid input type. Please provide numbers."
    else:
        return result
# Testing the function
print(divide_numbers(10, 2))  # Valid division
print(divide_numbers(10, 0))  # Division by zero
print(divide_numbers(10, 'a'))  # Invalid input type
# python file handling excercise
def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)
def read_from_file(filename):
    with open(filename, 'r') as file:
        return file.read()
# Writing to a file
write_to_file('example.txt', 'Hello, World!')
# Reading from the file
file_content = read_from_file('example.txt')
print(file_content)
# python module excercise
import math
def calculate_circle_area(radius):
    return math.pi * (radius ** 2)
# Testing the function
area = calculate_circle_area(5)
print(f"Area of the circle with radius 5: {area}")
