import sys
import boto3
import logging
from urllib.parse import urlparse
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import current_timestamp, date_format

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs are written to stdout
)
logger = logging.getLogger(__name__)

# Read job parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 's3_input_path', 'dynamodb_table_name'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Starting Glue job with arguments: %s", args)

s3_input_path = args['s3_input_path']
dynamodb_table_name = args['dynamodb_table_name']

# Check if the file exists in S3
def check_file_exists(s3_input_path):
    s3 = boto3.client('s3')
    parsed_url = urlparse(s3_input_path)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        logger.info("File found in S3: %s", s3_input_path)
        return True
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.warning("File not found in S3: %s", s3_input_path)
        else:
            logger.error("Error checking file in S3: %s", str(e))
        return False

if not check_file_exists(s3_input_path):
    logger.warning("Exiting job as file is not available to process.")
    job.commit()
    sys.exit(0)

logger.info("Reading CSV from S3: %s", s3_input_path)

# Read CSV
df = spark.read.csv(s3_input_path, header=True)

logger.info("Records loaded: %d", df.count())

# Cast schema + add CreateDate
df = (
    df.withColumn("Id", df["ID"].cast("string"))
      .withColumn("Name", df["Name"].cast("string"))
      .withColumn("Age", df["Age"].cast("int"))
      .withColumn("Address", df["Address"].cast("string"))
      .withColumn("CreateDate", date_format(current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss"))
)

# Convert to DynamicFrame
dyf = DynamicFrame.fromDF(df, glueContext, "dyf")

logger.info("Writing to DynamoDB table: %s", dynamodb_table_name)

# Write to DynamoDB using Glue sink
glueContext.write_dynamic_frame_from_options(
    frame=dyf,
    connection_type="dynamodb",
    connection_options={
        "dynamodb.region": "us-west-2",
        "dynamodb.output.tableName": dynamodb_table_name,
        "dynamodb.throughput.write.percent": "1.0"
    }
)

logger.info("Data successfully written to DynamoDB!")

# Move the file to an archive location
def move_to_archive(s3_input_path, archive_path):
    s3 = boto3.client('s3')
    parsed_url = urlparse(s3_input_path)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    archive_parsed_url = urlparse(archive_path)
    archive_bucket = archive_parsed_url.netloc
    archive_key = archive_parsed_url.path.lstrip('/')
    
    # Copy the file to the archive location
    s3.copy_object(
        Bucket=archive_bucket,
        CopySource={'Bucket': bucket_name, 'Key': key},
        Key=archive_key
    )
    # Delete the original file
    s3.delete_object(Bucket=bucket_name, Key=key)
    logger.info("File moved to archive: %s", archive_path)

# Define the archive path
archive_path = s3_input_path.replace("input/", "archive/")

move_to_archive(s3_input_path, archive_path)

job.commit()
logger.info("Glue job completed successfully.")