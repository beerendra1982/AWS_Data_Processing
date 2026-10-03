import boto3
import csv
from datetime import datetime
from botocore.exceptions import ClientError

# -----------------------------
# 1. Create DynamoDB Table
# -----------------------------
dynamodb = boto3.client('dynamodb', region_name='eu-west-2')
table_name = "BeeruTable"

try:
    existing_tables = dynamodb.list_tables()["TableNames"]
    if table_name in existing_tables:
        print(f"Table {table_name} already exists.")
    else:
        response = dynamodb.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "Id", "AttributeType": "S"}
            ],
            KeySchema=[
                {"AttributeName": "Id", "KeyType": "HASH"}
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5
            }
        )
        print(f"Creating table {table_name}...")
        dynamodb.get_waiter('table_exists').wait(TableName=table_name)
        print("Table created successfully.")
except ClientError as e:
    print(f"Error: {e.response['Error']['Message']}")

# -----------------------------
# 2. Insert CSV Data
# -----------------------------
dynamodb_resource = boto3.resource('dynamodb', region_name='eu-west-2')
table = dynamodb_resource.Table(table_name)

csv_file_path = "/Users/beerendrasingh/Desktop/AWS_Data_Processing/pyspark_aws_dynamodb/emp_details.csv"

with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)

    for row in reader:
        item = {
            "Id": row["ID"],
            "Name": row["Name"],
            "Age": int(row["Age"]),
            "Address": row["Address"],
            "CreateDate": datetime.now().isoformat()
        }

        table.put_item(Item=item)
        print(f"Inserted: {item}")

print("All CSV rows inserted successfully!")
