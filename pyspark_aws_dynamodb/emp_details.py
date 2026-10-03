import csv
import random
import string

# Function to generate random names
def random_name():
    return ''.join(random.choices(string.ascii_letters, k=random.randint(5, 10)))

# Function to generate random addresses
def random_address():
    return ''.join(random.choices(string.ascii_letters + string.digits + " ", k=random.randint(10, 20)))

# File path
file_path = '/Users/beerendrasingh/Desktop/AWS_Data_Processing/pyspark_aws_dynamodb/emp_details.csv'

# Generate 1000 records
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Name', 'Age', 'Address'])  # Write header
    
    for i in range(1, 1001):
        writer.writerow([
            i,  # ID
            random_name(),  # Name
            random.randint(18, 65),  # Age
            random_address()  # Address
        ])

print(f"CSV file generated at {file_path}")