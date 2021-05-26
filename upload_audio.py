from __future__ import unicode_literals
import os


import os
import io

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/dorjzodovs.batjargal/Desktop/Personal/Speech2Text/Speech2Text_backend/key/top-campaign-313812-50890e9f72d0.json"

# from gcloud import storage

from google.cloud import storage

client = storage.Client()

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

