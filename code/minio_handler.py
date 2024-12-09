import json
from minio import Minio
from minio.error import S3Error


class MinioHandler:
    def __init__(self, endpoint: str, username: str, password: str, bucket_name: str):
        self.client = Minio(
            endpoint,
            access_key=username,
            secret_key=password,
            secure=False
        )
        self.bucket_name = bucket_name
        self._create_bucket()

    def _create_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created.")
            else:
                print(f"Bucket '{self.bucket_name}' already exists.")
        except S3Error as e:
            print(f"Error creating the bucket: {e}")

    def upload_json(self, object_name: str, json_data: dict):
        try:
            temp_file = f"/tmp/{object_name}"
            with open(temp_file, "w") as f:
                json.dump(json_data, f, indent=4)
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=temp_file,
                content_type="application/json"
            )
            print(f"JSON file successfully uploaded: {object_name}")
        except S3Error as e:
            print(f"Error uploading the JSON file: {e}")


