from flask import Flask, Response
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import os

# Initialize Flask application
app = Flask(__name__)

# Constants for model and label map paths
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'C:/Users/vidya/OneDrive/Desktop/Proj/models/research/object_detection/data/mscoco_label_map.pbtxt'

# Check if model file exists, and download if necessary
if not os.path.exists(PATH_TO_CKPT):
    import urllib.request
    import tarfile

    print("Model file not found. Downloading...")
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

    # Extract the model
    with tarfile.open(MODEL_FILE) as tar:
        tar.extractall()
    print("Model downloaded and extracted.")

# Load the TensorFlow object detection model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load the label map
if not os.path.exists(PATH_TO_LABELS):
    raise FileNotFoundError("Label map not found. Please ensure it's at the correct path.")

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Generator function for video frames with object detection
def gen_frames():
    cap = cv2.VideoCapture(0)  # Open default webcam
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Expand dimensions to meet model requirements
                image_np_expanded = np.expand_dims(np.array(frame), axis=0)

                # Fetch required tensors for object detection
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Perform object detection
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )

                # Visualize detected objects on the frame
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                )

                # Encode the frame for HTTP streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Define Flask routes
@app.route('/')
def index():
    return "Welcome to the Object Detection Video Stream!"

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace;boundary=frame')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
