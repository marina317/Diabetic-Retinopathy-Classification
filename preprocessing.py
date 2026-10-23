import cv2 as cv
def load_image(image_path):
    image = cv.imread(image_path)
    return image

def green_channel_selection(image):
    blue, green, red = cv.split(image)
    return green


def run_preprocessing(image_path):
    image = load_image(image_path)
    green = green_channel_selection(image)
    return green
