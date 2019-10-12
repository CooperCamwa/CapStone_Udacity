from styx_msgs.msg import TrafficLight
import tensorflow as tf
import cv2
import numpy as np
import os
import rospy

def to_bb(box):

    bb = {}

    bb["y0"] = box[0] 
    bb["x0"] = box[1] 
    bb["y1"] = box[2] 
    bb["x1"] = box[3]

    return bb
         
    #  box = [int(box[0] * dim[0]), int(box[1] * dim[1]), int(box[2] * dim[0]), int(box[3] * dim[1])]
    #                 box_h, box_w = (box[2] - box[0], box[3] - box[1])
                   
def draw_bb(frame,bb,col=(255,0,0),th=2):

    x0 = int(bb["x0"])
    x1 = int(bb["x1"])
    y0 = int(bb["y0"])
    y1 = int(bb["y1"])

    frame = cv2.line(frame, (x0,y0),(x1,y0),col, th)
    frame = cv2.line(frame, (x0,y1),(x1,y1),col, th)
    frame = cv2.line(frame, (x0,y0),(x0,y1),col, th)
    frame = cv2.line(frame, (x1,y0),(x1,y1),col, th)

    return frame

class TLClassifier(object):
    def __init__(self):
        
        
        self.dg = tf.Graph()

        self.model_fn = "frozen_inference_graph.pb"
        #self.model_fn = "frozen_inference_graph_aakarsh_01.pb"

        pwd = os.path.dirname(os.path.realpath(__file__))
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(pwd+"/models/"+ self.model_fn, 'rb') as f:
                gdef.ParseFromString(f.read())
                tf.import_graph_def(gdef, name="")

            self.session = tf.Session(graph=self.dg)
            # The name of the tensor is in the form tensor_name:tensor_index
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.dg.get_tensor_by_name('num_detections:0')  

    def detect_regions(self, image):

        with self.dg.as_default():

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            tf_image_input = np.expand_dims(image, axis=0)

            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            ret = []
            detection_threshold = 0.5

            # Traffic signals are labelled 10 in COCO
            for idx, cl in enumerate(detection_classes.tolist()):
                if cl == 10:
                    if detection_scores[idx] < detection_threshold:
                        continue
                    dim = image.shape[0:2]
                    box = detection_boxes[idx]
                    box = [int(box[0] * dim[0]), int(box[1] * dim[1]), int(box[2] * dim[0]), int(box[3] * dim[1])]
                    box_h, box_w = (box[2] - box[0], box[3] - box[1])
                    if box_h / box_w < 1.6:
                        continue
                    #print('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                    ret.append(box)
        return ret

    def mark_up(self,boxes,image):

        for det in boxes:
            
            box,cl = det
            bb = to_bb(box)

            if cl == 1:
                draw_bb(image,bb,col=(0,255,0),th=4)
            elif cl == 2:
                draw_bb(image,bb,col=(255,255,0),th=4)
            elif cl == 3: 
                draw_bb(image,bb,col=(0,0,255),th=4)
            else: 
                draw_bb(image,bb,col=(255,255,255),th=4)

        return image   

    def vote(self,detections):

        # 1: Green
        # 2: Yellow
        # 3: Red

        cl_count = {}

        if len(boxes) <= 0:
            return TrafficLight.UNKNOWN

        for det in detections:
            box,cl = det
            if cl in cl_count:
                cl_count[cl] += 1
            else:     
                cl_count[cl] = 0

        best_cl = -1
        max_ct = -1

        for k in cl_count:
            if cl_count[k]  > max_ct:
                max_ct = cl_count[k]
                best_cl = k

        if best_cl == 1:
            return TrafficLight.GREEN
        elif best_cl == 2:
            return TrafficLight.YELLOW
        elif best_cl == 3:
            return TrafficLight.RED
        else:
            rospy.logerr("best cl is unknown %s",best_cl)
            return TrafficLight.UNKNOWN

    def patch_classifier(box,img):

        class_image = cv2.resize(img[box[0]:box[2], box[1]:box[3]], (32, 32))
            # The green needs to be checked first since red appears in many other components
            # For example traffice signs / other colors
        ret_green, thresh_green = cv2.threshold(class_image[:, :, 1], 170, 255, cv2.THRESH_BINARY)
        green_count = cv2.countNonZero(thresh_green)
        #print('green_count', green_count)
        box_h, box_w = (box[2] - box[0], box[3] - box[1])
        img_size = 32 * 32
        #print('img_size', 32 * 32)

        ret_red, thresh_red = cv2.threshold(class_image[:, :, 0], 170, 255, cv2.THRESH_BINARY)
        red_count = cv2.countNonZero(thresh_red)
        #print('red_count', red_count)
        #if green_count < 0.1 * img_size and red_count < 0.1 * img_size:
        #    rospy.logerr('YELLOW')
        #    return TrafficLight.YELLOW
        #else:
        if red_count > green_count:
            rospy.logerr('RED')
            return TrafficLight.RED
        else:
            rospy.logerr('GREEN')
            return TrafficLight.GREEN


    def get_classification_nikhil(self,boxes,img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #rospy.logerr("got image")
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        if boxes is None:
            rospy.logerr("Couldn't locate lights")
            return TrafficLight.UNKNOWN
        i = 0

        for det in boxes:
            


        return TrafficLight.UNKNOWN            
