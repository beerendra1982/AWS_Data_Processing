import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
table = dynamodb.Table('BeeruTable')

response = table.get_item(
    Key={
        "Id": "3"
    }
)

if "Item" in response:
    print("Record found:", response["Item"])
else:
    print("No record found for Id = 1")
