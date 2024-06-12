import os
import io
from PIL import Image, ImageOps
from google.cloud import vision
from google.oauth2 import service_account
import boto3
import json

region = 'ap-northeast-2'
bucket_name = 'capstone-almaeng2'
folder_name = 'todocr/test'
testdown_dir = "/tmp/todocr/test"
test_dir = "/tmp/todocr/test"
credentials_path='./servicekey.json'

class TimeOfDayRecognizer:
    def __init__(self, testdown_dir, test_dir,credentials_path):
        self.testdown_dir = testdown_dir
        self.test_dir = test_dir
        self.tod = ['아침', '점심', '저녁']
        self.credentials_path = credentials_path
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)

    def rotate_images(self, image, num):
        if num == 0:
            return image
        elif num == 1:
            return ImageOps.mirror(image.rotate(180))
        elif num == 2:
            return ImageOps.mirror(image)
        elif num == 3:
            return image.rotate(90, expand=True)
        else:
            return image.rotate(270, expand=True)
        
    def findtod_ee(self, tod_ee, simple_results_ee):
        count = 0
        for t, r in zip(tod_ee, simple_results_ee):
            if t == r:
                count += 1
        if count >= len(tod_ee) * 0.5:
            return tod_ee

    def findtod(self, simple_results_):
        result = []
        for tod in self.tod:
            for simple_result in simple_results_:
                foundtod = self.findtod_ee(tod, simple_result)
                if foundtod:
                    result.append(foundtod)
                    result=result[0]
        return result if result else "None"

    def recognize_time_of_day(self):
        files = os.listdir(self.test_dir)

        if not files:
            raise FileNotFoundError("No images found in the test directory.")

        image_path = os.path.join(self.test_dir, files[0])

        print(f"Processing image: {image_path}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        org_image = Image.open(image_path)
        image_rgb = org_image.convert("RGB")
        
        num = 0
        while True:
            image_list = self.rotate_images(image_rgb, num)

            buffer = io.BytesIO()
            image_list.save(buffer, format='JPEG')
            content = buffer.getvalue()
            
            vision_image = vision.Image(content=content)
            response = self.client.text_detection(image=vision_image)
            texts = response.text_annotations
            detected_text = texts[0].description
            ocr_result = self.findtod(detected_text)

            if ocr_result != 'None':
                return ocr_result

            num += 1
            if num > 4:  # Exit the loop if all rotations have been tried
                break

        return "None"


def connect_to_s3(self):
    # Connect to S3
    self.s3 = boto3.client('s3')

def download_images_from_s3(self, bucket_name, folder_name):
    # Ensure the directory exists
    os.makedirs(self.test_dir, exist_ok=True)
    
    # Clear the directory
    files = os.listdir(self.test_dir)
    for file in files:
        img_dir = os.path.join(self.test_dir, file)
        os.remove(img_dir)
    
    # List objects in the specified S3 bucket and folder
    response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    for obj in response.get('Contents', []):
        key = obj['Key']
        if not key.endswith('/'):  # If it's not a folder
            local_file_path = os.path.join(self.testdown_dir, os.path.basename(key))
            self.s3.download_file(bucket_name, key, local_file_path)
            print(f"{key} downloaded to: {local_file_path}")

def handler(event, context):

    os.makedirs(testdown_dir, exist_ok=True)
    tod_recognizer = TimeOfDayRecognizer(testdown_dir, test_dir,credentials_path)
    connect_to_s3(tod_recognizer)
    download_images_from_s3(tod_recognizer, bucket_name, folder_name)
    result = tod_recognizer.recognize_time_of_day()

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({'result': result}, ensure_ascii=False)
    }

# Comment out the direct call to handler
# handler('e','t')
