import pytesseract   # https://github.com/madmaze/pytesseract
import cv2
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

MAX_TEMP = 1000
MIN_TEMP = 0
INTENSITY_RANGE = [55, 255]    # this is the corresponding grey scale intensity range
MAX_TEMP_LOCATION = [70, 100, 500, 600]    # this corresponding to the fixed location of the highest temperature read at the scale
MIN_TEMP_LOCATION = [390, 420, 500, 600]   # this corresponding to the fixed location of the lowest temperature read at the scale

def clean_ocr_output(outputs):
    """ clean the OCR reading output
    outputs: the temperature scale numbers read from thermal image using OCR
    """
    nums = []
    for line in outputs.splitlines():
        clean_line = line.strip().split(".")[0].split(" ")[0]
        if clean_line.isdigit():
            nums.append(int(clean_line))
    if not nums:
        return []
    nums = sorted(nums) if min(nums) >= MIN_TEMP and max(nums) <= MAX_TEMP else []
    return nums

def get_temp_range_from_thermal(path_to_file):
    """ Read the temperature scale numbers from thermal image using OCR 
    path_to_file: the thermal image data file
    """
    image = cv2.imread(path_to_file, 0)

    # method 1: crop the fixed area and do the OCR
    min_temp_read = pytesseract.image_to_string(image[MIN_TEMP_LOCATION[0]:MIN_TEMP_LOCATION[1], MIN_TEMP_LOCATION[2]:MIN_TEMP_LOCATION[3]])
    max_temp_read = pytesseract.image_to_string(image[MAX_TEMP_LOCATION[0]:MAX_TEMP_LOCATION[1], MAX_TEMP_LOCATION[2]:MAX_TEMP_LOCATION[3]])
    combined_output = min_temp_read + max_temp_read

    # method 2: OCR based the whole picture
    outputs_without_crop = pytesseract.image_to_string(image)

    # clean the OCR output and return the combined result
    cleaned_output_1 = clean_ocr_output(combined_output)
    cleaned_output_2 = clean_ocr_output(outputs_without_crop)
    return [min(cleaned_output_1+cleaned_output_2), max(cleaned_output_1+cleaned_output_2)]

def calc_temps_in_bounding_box(img_file, bounding_box_range, set_minimum_temp=25):
    """ Transform the thermal image intensity to corresponding temperature, within the bounding box, and calculate the statistics.
    img_file: the thermal image file
    bounding_box_range: the bounding box area for the image
    set_minimum_temp: a pre-set minimum temperature based on the environment setting
    """
    bb_x_min, bb_x_max, bb_y_min, bb_y_max = bounding_box_range
    image = cv2.imread(img_file, 0)
    image_np = np.array(image)
    temp_scale = get_temp_range_from_thermal(img_file) # get the scale of the temperature

    # based on scale of the temperature and the intensity range, calculate the linear function
    try:
        slope = (temp_scale[1] - temp_scale[0]) / (INTENSITY_RANGE[1] - INTENSITY_RANGE[0])
    except:
        logging.info('An error when reading the temp scale')
        return None
    slope = (temp_scale[1] - temp_scale[0]) / (INTENSITY_RANGE[1] - INTENSITY_RANGE[0])
    intercept = temp_scale[1] - slope*INTENSITY_RANGE[1]

    # transform pixel intensity to the corresponding temperature, within the bounding box, with a pre-set minimum temp 
    pixel_intensity_in_bb = image_np[bb_x_min:bb_x_max, bb_y_min:bb_y_max].flatten()
    pixel_temp_in_bb = [int(intensity*slope) + intercept for intensity in pixel_intensity_in_bb]
    pixel_temp_stepwise_in_bb = [max(set_minimum_temp, temp) for temp in pixel_temp_in_bb] # 
    return (round(np.min(pixel_temp_stepwise_in_bb), 1), 
            round(np.max(pixel_temp_stepwise_in_bb), 1), 
            round(np.mean(pixel_temp_stepwise_in_bb), 1))

def drawBoundingBoxes(imageData, bounding_box, label, color=(255, 255, 255), thick=1):
    """ Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    imgHeight = bounding_box[1] - bounding_box[0]
    top, bottom, left, right= bounding_box
    cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
    cv2.putText(imageData, label, (left, top - 12), 0, 1e-2 * imgHeight, color, thick//2)
    return imageData
