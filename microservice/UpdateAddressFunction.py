import json
import boto3

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table('BeeruTable')

def lambda_handler(event, context):
    for record in event['Records']:
        message = json.loads(record['body'])
        print(f"Processing message: {message}")

        # Extract details from the message
        emp_id = message['id']
        current_address = message['current_address']
        action = message['action']

        if action == "update_address":
            # Perform the address update logic here
            try:
                # Update the address in DynamoDB
                response = table.update_item(
                    Key={
                        'Id': emp_id  # Ensure 'Id' matches the Partition Key name in DynamoDB
                    },
                    UpdateExpression="SET Address = :new_address",
                    ExpressionAttributeValues={
                        ':new_address': current_address + " _1"  # Append "_1" to the current address
                    },
                    ReturnValues="UPDATED_NEW"
                )

                print(f"Address updated for employee ID {emp_id}: {response['Attributes']}")
            except Exception as e:
                print(f"Error updating address for employee ID {emp_id}: {str(e)}")