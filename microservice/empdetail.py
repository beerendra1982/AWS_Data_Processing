import json
import boto3
import os

# Initialize DynamoDB and SQS clients
dynamodb = boto3.client('dynamodb', region_name='eu-west-2')
sqs = boto3.client('sqs', region_name='eu-west-2')

# Environment variables
TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "BeeruTable")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "https://sqs.eu-west-2.amazonaws.com/371456644911/UpdateAddressQueue")

def lambda_handler(event, context):
    try:
        # Log the event for debugging
        print("Event received:", json.dumps(event))

        # Extract the employee ID from the query parameters
        emp_id = event.get('queryStringParameters', {}).get('id', None)
        if not emp_id:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"status": "error", "message": "Missing 'id' in query parameters"})
            }

        # Query DynamoDB for the employee details
        response = dynamodb.get_item(
            TableName=TABLE_NAME,
            Key={
                'Id': {'S': emp_id}  # Ensure 'Id' matches the Partition Key name in DynamoDB
            }
        )

        # Check if the item exists
        if 'Item' in response:
            # Convert DynamoDB item to a JSON serializable format
            employee = {k: list(v.values())[0] for k, v in response['Item'].items()}

            # Send a message to the SQS queue to update the address
            sqs_message = {
                "id": emp_id,
                "current_address": employee.get("Address"),
                "action": "update_address"
            }
            sqs_response = sqs.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(sqs_message)
            )

            print(f"SQS message sent: {sqs_response['MessageId']}")

            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"status": "success", "data": employee, "sqs_message_id": sqs_response['MessageId']})
            }
        else:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"status": "error", "message": "Employee not found"})
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"status": "error", "message": str(e)})
        }