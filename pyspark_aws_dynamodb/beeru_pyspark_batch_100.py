from pyspark.sql import SparkSession
import boto3
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DynamoDB Integration") \
    .getOrCreate()

# Read the CSV file into a DataFrame
csv_file_path = "/Users/beerendrasingh/Desktop/AWS_Data_Processing/pyspark_aws_dynamodb/emp_details.csv"
df = spark.read.csv(csv_file_path, header=True)

# Table name
table_name = "BeeruTable"

# Function to write data to DynamoDB in batches of 100
def write_to_dynamodb_in_batches(rows, batch_size=100):
    try:
        # Initialize DynamoDB resource
        dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
        table = dynamodb.Table(table_name)

        # Split rows into chunks of batch_size
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i + batch_size]
            with table.batch_writer() as batch:
                for row in batch_rows:
                    batch.put_item(
                        Item={
                            "Id": row["ID"],  # Partition key
                            "Name": row["Name"],
                            "Age": int(row["Age"]),
                            "Address": row["Address"],
                            "CreateDate": datetime.now().isoformat()  # Default system datetime
                        }
                    )
            print(f"Batch {i // batch_size + 1} inserted successfully!")
    except Exception as e:
        print(f"Error during batch insert: {str(e)}")

# Collect all rows from the DataFrame and write to DynamoDB in batches
rows = df.collect()  # Collect all rows to the driver
write_to_dynamodb_in_batches(rows, batch_size=10)

print("All data successfully written to DynamoDB in batches!")