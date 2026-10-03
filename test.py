import boto3, json

client = boto3.client("bedrock-runtime", region_name="us-west-2")

payload = {
    "messages": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}]
        }
    ]
}

resp = client.invoke_model(
    modelId="amazon.nova-pro-v1",
    contentType="application/json",
    accept="application/json",
    body=json.dumps(payload)
)

print(resp["body"].read())
