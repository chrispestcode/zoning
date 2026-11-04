import boto3
from dotenv import load_dotenv
from aws.aws_s3_client import AwsS3Client


class AwsSession:
    def __init__(
        self,
        role_arn: str,
        region_name: str,
        session_name: str = 'AwsSession',
    ):
        load_dotenv()
        sts_client = boto3.client('sts', region_name=region_name)

        response = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name,
        )

        credentials = response['Credentials']

        self._session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=region_name,
        )

    def get_s3_client(self) -> AwsS3Client:
        return AwsS3Client(self._session.client('s3'))
    
if __name__ == "__main__":
    session = AwsSession(
        role_arn="arn:aws:iam::158179669543:role/s3-read-only",
        region_name="us-east-1")
    client = session.get_s3_client()
    bucket = client.list_buckets()[1]
    print(client.list_objects(bucket_name=bucket)['Contents'][0])
# object_aws = (client.get_object(bucket_name=bucket, object_name='11766-Provider-Networks-508.pdf'))
# with open('/home/tchrisp/projects/modern_search/logs/11766-Provider-Networks-508123.pdf', 'wb') as f:
#     f.write(object_aws)

