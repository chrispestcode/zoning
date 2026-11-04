import boto3

class AwsS3Client:
    def __init__(self, s3_client: boto3.client):
        self.s3_client = s3_client
        
    def upload_file(self, file_path: str, bucket_name: str, object_name: str):
        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name)
            print(f"File {file_path} uploaded to {bucket_name}/{object_name}")
        except Exception as e:
            print(f"Error uploading file: {e}")
    
    def download_file(self, bucket_name: str, object_name: str, file_path: str):
        try:
            self.s3_client.download_file(bucket_name, object_name, file_path)
            print(f"File {object_name} downloaded from {bucket_name} to {file_path}")
        except Exception as e:
            print(f"Error downloading file: {e}")
    
    def list_buckets(self):
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            return buckets
        except Exception as e:
            print(f"Error listing buckets: {e}")
            return []
    
    def list_objects(self, bucket_name: str):
        try:
            return self.s3_client.list_objects_v2(Bucket=bucket_name)
        except Exception as e:
            print(f"Error listing objects: {e}")
            return None  
    
    def get_object(self, bucket_name: str, object_name: str):
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_name)
            return response['Body'].read()
        except Exception as e:
            print(f"Error getting object: {e}")
            return None

    def put_object(self, bucket_name: str, object_name: str, data: bytes):
        try:
            self.s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=data)
            print(f"Object {object_name} put in bucket {bucket_name}")
        except Exception as e:
            print(f"Error putting object: {e}")