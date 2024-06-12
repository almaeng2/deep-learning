import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import boto3
import json

# Usage:
bucket_name = 'capstone-almaeng2'
folder_name = 'housepill/test/for test'
testdown_dir = "/tmp/housepill/test/for test"
base_model_dir = './Best_Model/tflite'
test_dir = "/tmp/housepill/test"
train_class_names = ["닥터베아정","뒷면","베아제정","신신파스아렉스","어린이부루펜시럽","어린이타이레놀현탁액","제일쿨파프","타이레놀정","판콜에이내복액","판피린티정","훼스탈골드정","훼스탈플러스정"]

class ModelPredictor:
    def __init__(self, bucket_name, folder_name,testdown_dir, base_model_dir, test_dir, train_class_names):
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        self.testdown_dir = testdown_dir
        self.base_model_dir = base_model_dir
        self.test_dir = test_dir
        self.train_class_names = train_class_names
    
    def connect_to_s3(self):
        self.s3=boto3.client('s3')

    def download_images_from_s3(self):
        files = os.listdir(self.test_dir)
        img_dir = os.path.join(self.test_dir, str(files[0]))
        img_files = os.listdir(img_dir)
        for f_path in img_files:
            final_path = os.path.join(img_dir, f_path)
            os.remove(final_path)

        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=self.folder_name)
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('/'):  # If it's not a folder
                local_file_path = os.path.join(self.testdown_dir, os.path.basename(key))
                self.s3.download_file(self.bucket_name, key, local_file_path)
                print(f"{key} downloaded to: {local_file_path}")

        batch_size = 32
        img_height = 128
        img_width = 128
        img_size = (img_height, img_width)

        # Collect images manually
        image_paths = glob.glob(os.path.join(self.test_dir, '*', '*.jpg'))
        images = []
        for img_path in image_paths:
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img)
            img = img / 255.0  # Rescale to 0-1
            images.append(img)

        images = np.array(images)
        labels = [os.path.basename(os.path.dirname(img_path)) for img_path in image_paths]

        # Initialize test dataset
        test_ds = tf.data.Dataset.from_tensor_slices((images, labels))
        test_ds = test_ds.batch(batch_size)

        model_files = os.listdir(self.base_model_dir)
        self.model_dir = [os.path.join(self.base_model_dir, model) for model in model_files if model.endswith('.tflite')]

        model_predicted_label = []

        for model_path in self.model_dir:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            predictions = []
            for images_batch, labels_batch in test_ds:
                # Resize images to match input shape of the model
                interpreter.resize_tensor_input(input_details[0]['index'], images_batch.shape)
                interpreter.allocate_tensors()

                # Set input tensor
                input_data = images_batch.numpy().astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                # Run inference
                interpreter.invoke()

                # Get output tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predictions.extend(output_data)

            real_predictions = np.argmax(predictions, axis=1)
            for i in real_predictions:
                model_predicted_label.append(i)

        model_predicted_label = np.array(model_predicted_label)
        model_predicted_label = np.ravel(model_predicted_label)

        max_per = []
        for i, class_name in enumerate(self.train_class_names):
            cnt = np.sum(model_predicted_label == i)
            percentage = (cnt / len(model_predicted_label)) * 100
            max_per.append(percentage)

        result_class_name = self.train_class_names[max_per.index(max(max_per))]
        if result_class_name=='뒷면':
            return "None"
        else:
            return result_class_name


def handler(event, context):

    os.makedirs(testdown_dir, exist_ok=True)
    model_predictor = ModelPredictor(bucket_name, folder_name, testdown_dir, base_model_dir, test_dir, train_class_names)
    model_predictor.connect_to_s3()
    result= model_predictor.download_images_from_s3()

    return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'result': result},ensure_ascii=False)
        }
