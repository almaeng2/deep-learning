import os
import numpy as np
import pandas as pd
from numpy import sqrt, sum, square
from PIL import Image
from rembg import remove
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
import boto3
import json
from ultralytics import YOLO

#Environment variables
# os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
# os.environ['YOLO_CONFIG_DIR'] = '/tmp/ultralytics'
# os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
# os.environ['U2NET_HOME'] = '/tmp/u2net'

# Constants

region='ap-northeast-2'
BUCKET_NAME = 'capstone-almaeng2'
FOLDER_NAME = 'similarity/test'
SAVE_DIR = "/tmp/image"
OUTPUT_FOLDER = "/tmp/rembg"


# Load models
feature_model = tf.keras.models.load_model('./VGG19.h5')
yolo_model = YOLO("./best.pt")

def download_from_s3(bucket_name, folder_name, save_dir):
    client = boto3.client('s3')
    for f in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, f))

    response = client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    for obj in response['Contents']:
        key = obj['Key']
        if not key.endswith('/'):
            file_name = os.path.basename(key)
            local_file_path = os.path.join(save_dir, file_name.replace('/', '_'))
            client.download_file(bucket_name, key, local_file_path)
            print(f"{key} downloaded to: {local_file_path}")

    new_size = (1080 , 1440)
    # Resize image
    for filename in os.listdir(save_dir):
        with Image.open(os.path.join(save_dir, filename)) as img:
                # Resize image
                img_resized = img.resize(new_size)
                # Save it to the output directory
                img_resized.save(os.path.join(save_dir, filename))

def crop_objects_from_image(image_path, results, save_path):
    try:
        image = Image.open(image_path)
    except IOError:
        print(f"Failed to load image: {image_path}")
        return

    for idx, result in enumerate(results):
        boxes = result.boxes.xyxy
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_img_path = os.path.join(save_path, f"{os.path.basename(image_path).split('.')[0]}_crop_{idx}_{i}.jpg")
            if cropped_img_path[-5] == "0":
                cropped_img_path2 = cropped_img_path[:-13] + '.jpg'
                cropped_img.save(cropped_img_path2)
                print(f"Cropped image saved to {cropped_img_path2}")

def vectorize_image(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)[0]
    return features / np.linalg.norm(features)

def calculate_euclidean_distance_similarity(vector1, vector2):
    euclidean_distance = sqrt(sum(square(vector1 - vector2)))
    return euclidean_distance

def process_images_and_calculate_similarity(save_dir, output_folder, feature_model, yolo_model):
    results = yolo_model.predict(save_dir)
    for result in results:
        image_path = result.path
        crop_objects_from_image(image_path, [result], save_dir)

    for f in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, f))

    for filename in os.listdir(save_dir):
        if filename.endswith((".jpg", ".png")):
            input_image_path = os.path.join(save_dir, filename)
            input_image = Image.open(input_image_path)
            output_image = remove(input_image)
            output_image_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            output_image.save(output_image_path)

    image_files = [f for f in os.listdir(output_folder)]
    input_vectors = [vectorize_image(os.path.join(output_folder, image_file), feature_model) for image_file in image_files]
    return input_vectors

def calculate_similarity_and_find_pill(input_vectors):
    mean_df = pd.read_csv('./pills_image_vector.csv', index_col=0)
    mean_arr = mean_df.values
    input_arr = np.array(input_vectors)

    similarity = []
    for mean_vector in mean_arr:
        for input_vector in input_arr:
            similarity.append(calculate_euclidean_distance_similarity(mean_vector, input_vector))

    similarity_df = pd.DataFrame(np.array(similarity).reshape(len(mean_arr), len(input_arr)))
    similarity_df['Mean'] = similarity_df.mean(axis=1)

    num_pills = similarity_df.shape[0] // len(input_arr)
    similarity_mean = similarity_df['Mean'].values.reshape(num_pills, -1).mean(axis=1)
    similarity_mean_df = pd.DataFrame({'Similarity': similarity_mean}, index=['Cough', 'Fatak', 'Lebrocol', 'Methylon', 'Retonase', 'Samnam', 'Siwonna', 'Suspen', 'Youngil'])

    min_index = similarity_mean_df['Similarity'].idxmin()
    return min_index

def change_name(pill_name):    
    if (pill_name=='Cough'):
        pill_name='코푸정'
    if (pill_name=='Fatak'):
        pill_name='파탁정20mg(파모티딘)'
    if (pill_name=='Lebrocol'):
        pill_name='레브로콜정60밀리그램(레보드로프로피진)'
    if (pill_name=='Methylon'):
        pill_name='메치론정(메틸프레드니솔론)'
    if (pill_name=='Retonase'):
        pill_name='레토나제정'
    if (pill_name=='Samnam'):
        pill_name='삼남아세트아미노펜정'
    if (pill_name=='Siwonna'):
        pill_name='시원나정(록소프로펜나트륨수화물)'
    if (pill_name=='Suspen'):
        pill_name='써스펜8시간이알서방정650mg(아세트아미노펜)'
    if (pill_name=='Youngil'):
        pill_name='영일클래리스로마이신정250mg'
    return pill_name

def handler(event, context):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    download_from_s3(BUCKET_NAME, FOLDER_NAME, SAVE_DIR)
    input_vectors = process_images_and_calculate_similarity(SAVE_DIR, OUTPUT_FOLDER, feature_model, yolo_model)
    pill_name = calculate_similarity_and_find_pill(input_vectors)
    result=change_name(pill_name)

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'result': result}, ensure_ascii=False)
    }