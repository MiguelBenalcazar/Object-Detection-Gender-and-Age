import numpy as np
import os

CURRENT_PATH = os.path.dirname(__file__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #Avoid to acces to the GPU

import cv2
import tensorflow as tf
import six
import numpy as np
#Import the object detection module.
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

INFERENCE_OBJECT_DETECTION = 0.75

#Patches: Updating tensorflow version 1.x to tensorflow 2.0
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
padding = 20


def load_model(model_name):
    model_dir = os.path.join(CURRENT_PATH, "models")
    model_dir = os.path.join(model_dir, model_name)
    model_dir = os.path.join(model_dir, "saved_model")
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model

def load_model_age_gender():
    model_dir = os.path.join(CURRENT_PATH, "models")
    model_dir = os.path.join(model_dir, "face_gender_age")

    faceProto = os.path.join(model_dir, "opencv_face_detector.pbtxt")
    faceModel = os.path.join(model_dir, "opencv_face_detector_uint8.pb")
    ageProto = os.path.join(model_dir,"age_deploy.prototxt")
    ageModel = os.path.join(model_dir,"age_net.caffemodel")
    genderProto = os.path.join(model_dir,"gender_deploy.prototxt")
    genderModel = os.path.join(model_dir, "gender_net.caffemodel")
    
    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    
    return faceNet, ageNet, genderNet


# List of the strings that is used to add correct label for each box.
labels = 'mscoco_label_map.pbtxt'
# labels = 'mscoco_complete_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(CURRENT_PATH, "data")
PATH_TO_LABELS = os.path.join(PATH_TO_LABELS, labels)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)




#https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# model_name = 'ssd_mobilenet_v1_coco_2018_01_28' 
# model_name = 'faster_rcnn_resnet50_coco_2018_01_28'
model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
# model_name = 'faster_rcnn_inception_v2_coco_2018_01_28'
# model_name = 'output_inference_graph_v1' #mi own model test
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def highlightFace(net, frame, conf_threshold=0.7):
    # frameOpencvDnn = []
    
    if frame is  None:
        return

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]


    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    

    net.setInput(blob)
    detections=net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

faceNet, ageNet, genderNet = load_model_age_gender()
cap = cv2.VideoCapture(0)
# labels = objectDetection_readLabels(path)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    output_dict = run_inference_for_single_image(detection_model, frame)
    
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
                

    for faceBox in faceBoxes:
        face = frame[max(0,faceBox[1]-padding): min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
                    
       
        
        cv2.putText(resultImg, f'{gender}, {age}', ( faceBox[0], faceBox[1] ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 1, cv2.LINE_AA)

    cv2.imshow('res',resultImg)
    
    # if output_dict['num_detections'] is not None and not 0:
    #     classes = output_dict['detection_classes']
    #     boxes = output_dict['detection_boxes']
    #     score = output_dict['detection_scores']
        
    #     for i in range(output_dict['num_detections']):
            
    #         if score[i] >= INFERENCE_OBJECT_DETECTION:
                
    #             if classes[i] == 1:
    #             # person  = [boxes[i][0] * frame.shape[0],  boxes[i][1] * frame.shape[0] boxes[i][2] * frame.shape[1] boxes[i][3] * frame.shape[1]]
            
    #                 box = tuple(boxes[i].tolist())
    #                 ymin, xmin, ymax, xmax = box
    #                 ymin, xmin, ymax, xmax = int(ymin * frame.shape[0]), int(xmin * frame.shape[1]), int(ymax * frame.shape[0]), int(xmax * frame.shape[1])
    #                 person = frame[max(0,ymin-padding): min(ymax + padding, frame.shape[0]-1),max(0,xmin-padding):min(xmax+padding, frame.shape[1]-1)]
               
                
    #                 resultImg, faceBoxes = highlightFace(faceNet, person)


    #                 if not faceBoxes:
    #                     print("No face detected")
                

    #                 for faceBox in faceBoxes:
    #                     face = person[max(0,faceBox[1]-padding): min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

    #                     blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    #                     genderNet.setInput(blob)
    #                     genderPreds = genderNet.forward()
    #                     gender = genderList[genderPreds[0].argmax()]
    #                     print(f'Gender: {gender}')

    #                     ageNet.setInput(blob)
    #                     agePreds = ageNet.forward()
    #                     age = ageList[agePreds[0].argmax()]
    #                     print(f'Age: {age[1:-1]} years')
                    
    #                     cv2.imshow(str(i), face)
    #                     cv2.rectangle(frame, ( xmin + faceBox[0] - padding, ymin + faceBox[1] - padding), ( xmin + faceBox[2] - padding,  ymin + faceBox[3] - padding ), color=(0, 255 , 255))
    #                     cv2.putText(frame, f'{gender}, {age}', (xmin + faceBox[0] - padding -10, ymin + faceBox[1] - padding -10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv2.LINE_AA)

                 

    
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates = True,
        line_thickness=2)


    # print(output_dict['detection_boxes'], output_dict['detection_scores'])
     # Display the resulting frame
    frame = cv2.resize(frame , (1200, 900))
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
