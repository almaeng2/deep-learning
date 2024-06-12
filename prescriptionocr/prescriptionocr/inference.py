import os
import re
from PIL import Image, ImageOps
from google.cloud import vision
from google.oauth2 import service_account
import boto3
import json
import io

bucket_name = 'capstone-almaeng2'
folder_name = 'prescription/test/'
testdown_dir = "/tmp/test/"
test_dir = "/tmp/test"
credentials_path = "./servicekey.json"

class PrescriptionReader:
    def __init__(self, origin_med, test_dir,testdown_dir, credentials_path):
        self.origin_med = origin_med
        self.test_dir=test_dir
        self.testdown_dir=testdown_dir
        
        self.credentials_path = credentials_path

        # Initialize the Vision API client with the key file
        self.credentials = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
    def connect_to_s3(self):
        self.s3 = boto3.client('s3')

    def download_images_from_s3(self, bucket_name, folder_name):
        files = os.listdir(self.test_dir)
        for f_path in files:
            final_path = os.path.join(self.test_dir, f_path)
            os.remove(final_path)

        response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('/'):
                file_name = os.path.basename(key)
                local_file_path = os.path.join(self.testdown_dir, file_name.replace('/', '_'))
                self.s3.download_file(bucket_name, key, local_file_path)
                print(f"{key} downloaded to: {local_file_path}")

    def remove_units(self):
        unit_patterns = [
            r'\d+mg',
            r'\d+밀리그램',
            r'\([^)]*\)'
        ]
        pattern = '|'.join(unit_patterns)
        self.med = [re.sub(pattern, '', med).strip() for med in self.origin_med]

    def rotate_image(self):
        files=os.listdir(self.test_dir)
        image_path=os.path.join(self.test_dir,files[0])
        image = Image.open(image_path)
        image_rgb = image.convert("RGB")

        self.image_list = [
            image_rgb,
            image_rgb.rotate(180, expand=True),
            image_rgb.rotate(90, expand=True),
            image_rgb.rotate(270, expand=True),
            ImageOps.mirror(image_rgb)
        ]

    def counting_number_of_words(self, med, text_split):
        l_med = len(med)
        cnt = 0
        for word in med:
            if word in text_split:
                cnt += 1
        return cnt <= l_med

    def findmed(self, med, text_split):
        count = 0
        count2 = 0
        for m, med_word in enumerate(med):
            for t, text_word in enumerate(text_split):
                if med_word == text_word:
                    if m == t:
                        count += 1
                    count2 += 1
        if count >= len(med) * 0.7 or count2 >= len(med) * 0.9:
            return med
        return None

    def ocr_and_extract(self):
        final = []
        for image in self.image_list:
            # Convert the image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            content = buffer.getvalue()

            # Perform text detection using Google Cloud Vision API
            vision_image = vision.Image(content=content)
            response = self.client.text_detection(image=vision_image)
            texts = response.text_annotations

            if not texts:
                continue

            # Extract detected text
            detected_text = texts[0].description
            text_split = detected_text.split()
            result = []
            for m in range(len(self.med)):
                for t in range(len(text_split)):
                    if self.counting_number_of_words(self.med[m], text_split[t]):
                        foundmed = self.findmed(self.med[m], text_split[t])
                        if foundmed is not None:
                            result.append(foundmed)
            result2 = []
            for i in result:
                for j in self.origin_med:
                    if i in j:
                        result2.append(j)
                        final.append(j)
            result2 = set(result2)

        if final:
            return set(final)
        else:
            return 'None'


# Example usage
origin_med = ['써스펜8시간이알서방정650mg(아세트아미노펜)', '코푸정', '시원나정(록소프로펜나트륨수화물)', '메치론정(메틸프레드니솔론)',
              '영일클래리스로마이신정250mg', '레브로콜정60밀리그램(레보드로프로피진)', '삼남아세트아미노펜정', '레토나제정',
              '파탁정20mg(파모티딘)']

def handler(event, context):
    os.makedirs(testdown_dir, exist_ok=True)
    prescription_reader = PrescriptionReader(origin_med, test_dir,testdown_dir, credentials_path)
    prescription_reader.connect_to_s3()
    prescription_reader.download_images_from_s3(bucket_name,folder_name)
    prescription_reader.remove_units()
    prescription_reader.rotate_image()
    result = prescription_reader.ocr_and_extract()
    #result=print(type(result))

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps({'result': list(result)}, ensure_ascii=False)
    }

# print(handler('e','t'))

