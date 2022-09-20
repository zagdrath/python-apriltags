import cv2
from pupil_apriltags import Detector

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 0)


def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))

    image = cv2.line(image, (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]), color, 3)
    image = cv2.line(image, (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH), color, 3)
    return image


def drawBox(image, point_a, point_b, point_c, point_d, corners, color):
    image = cv2.line(image, point_a, point_b, color, 3)
    image = cv2.line(image, point_b, point_c, color, 3)
    image = cv2.line(image, point_c, point_d, color, 3)
    image = cv2.line(image, point_d, point_a, color, 3)


def drawText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)

    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


# Setup tag detector
tag_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

# Setup video capture
camera_stream = cv2.VideoCapture(0)

is_looping = True

while is_looping:
    result, image = camera_stream.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check for tags
    detections = tag_detector.detect(gray_image)
    if not detections:
        print("WARNING: No Tags Found")
    else:
        # Reports and updates the camera stream if tags were found
        for detect in detections:
            print("Tag #: %s, Tag Center: %s" %
                  (detect.tag_id, detect.center))
            image = plotPoint(image, detect.center, CENTER_COLOR)
            image = drawText(image, detect.center, CENTER_COLOR, detect.tag_id)

            for corner in detect.corners:
                corners = (point_a, point_b, point_c, point_d)

                point_a = (int(point_a[0]), int(point_a[1]))
                point_b = (int(point_b[0]), int(point_b[1]))
                point_c = (int(point_c[0]), int(point_c[1]))
                point_d = (int(point_d[0]), int(point_d[1]))

                image = drawBox(image, point_a, point_b, point_c,
                                point_d, corners, CENTER_COLOR)

    # Update camera stream
    cv2.imshow('Camera Stream', image)

    # Wait for system to loop
    wait_key = cv2.waitKey(100)

    # Terminate loop if 'Return' key is hit
    if wait_key == 13:
        is_looping = False

# Clean up and dump last frame
cv2.destroyAllWindows()
cv2.imwrite("final.png", image)
