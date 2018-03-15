//
// Created by Andrei Nechaev on 3/14/18.
//

#pragma once

/* Errors */
#define CAMERA_NOT_AVAILABLE 1100

#define MODEL_NOT_AVAILABLE 1200

#define EYE_HAAR_CASCADE_NOT_AVAILABLE 1300;

/* General usage constants */
#define CONFIDENCE_LEVEL 0.93

/* Paths */
#define FACE_DNN_PROTO "models/face/deploy.prototxt.txt"
#define FACE_DNN_MODEL "models/face/res10_300x300_ssd_iter_140000.caffemodel"
#define EYE_HAAR_CASCADE "models/eye/haarcascade_eye.xml"