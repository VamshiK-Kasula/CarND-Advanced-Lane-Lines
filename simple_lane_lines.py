import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    # """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # """
    # Applies an image mask.

    # Only keeps the region of the image defined by the polygon
    # formed from `vertices`. The rest of the image is set to black.
    # `vertices` should be a numpy array of integer points.
    # """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def cal_y_intercept(x1, y1, x2, y2, slope):
    # """Function to calculate y-intercept at x=0"""
    y_intercept = y1 - (slope * x1)
    return y_intercept


def cal_x_intercept(x1, y1, x2, y2, img_height):
    # """Function to calculate x-intercept at y = image_height(540 pixels)"""
    slope = find_slope(x1, y1, x2, y2)
    y_intercept = cal_y_intercept(x1, y1, x2, y2, slope)
    x_intercept = (img_height - y_intercept)/slope
    return x_intercept


def find_slope(x1, y1, x2, y2):
    # """Function to determine the slope of a given line"""
    slope = (y2-y1)/(x2-x1)
    return slope


def separate_lines(lines, img_height, img_width):
    # """Function to group the lines from Hough transform output to left and right groups based on their slopes"""
    left_lines = {}
    right_lines = {}
    for line in lines:
        for x1, y1, x2, y2 in line:
            if 0 < cal_x_intercept(x1, y1, x2, y2, img_height) < img_width:
                slope = find_slope(x1, y1, x2, y2)
                if slope < 0:
                    left_lines[round(slope, 4)] = line
                else:
                    right_lines[round(slope, 4)] = line
    return left_lines, right_lines


def line_near_avg_slope(lane_lines):
    # """In this function, a line whose slope is closer to the average slope of the respective group of lines(left/right) is determined"""
    slope_data = list(lane_lines.keys())
    slope_avg = np.mean(slope_data)
    line_close_to_avg_slope = list(lane_lines.keys())[0]
    diff_to_avg_slope = abs(line_close_to_avg_slope-slope_avg)

    for key in list(lane_lines):
        slope_diff = abs(key-slope_avg)
        if (diff_to_avg_slope >= slope_diff):
            diff_to_avg_slope = slope_diff
            line_close_to_avg_slope = key
    return lane_lines.get(line_close_to_avg_slope)


def filter_lines(lane_lines):
    # """Removing outliners based on slope"""
    filtered_lines = []
    slope_data = list(lane_lines.keys())
    slope_avg = np.mean(slope_data)
    slope_std = np.std(slope_data)

    for key in list(lane_lines):
        if (abs(key-slope_avg) > 2*slope_std):
            lane_lines.pop(key)
    for key, value in lane_lines.items():
        filtered_lines.append(value)
    return lane_lines


def find_min_y_value(lines_with_slopes, img_width):
    # """In this function, minimum y-coordinate among the given lines is determined"""
    min_y = img_width
    lines = [values for values in lines_with_slopes.values()]
    for line in lines:
        for x1, y1, x2, y2 in line:
            min_y = min(min_y, y1, y2)
    return min_y


def extrapolate_lane(filtered_lines, img_height):
    # """In this function, the detemined lane lines are extended from along y-axis from image_height to minimum y-cordinate from the respective groups"""
    extrapolated_lane = []

    # Calculate the least y-cordinate from the group of lines and thus find the respective y-coordinate
    line_near_avg = line_near_avg_slope(filtered_lines)
    min_y = find_min_y_value(filtered_lines, img_height)
    x_intercept_at_min_y = cal_x_intercept(line_near_avg.item((0, 0)), line_near_avg.item(
        (0, 1)), line_near_avg.item((0, 2)), line_near_avg.item((0, 3)), min_y)

    # calculating the x-intercept at y = image_height
    x_intercept = cal_x_intercept(line_near_avg.item((0, 0)), line_near_avg.item(
        (0, 1)), line_near_avg.item((0, 2)), line_near_avg.item((0, 3)), img_height)

    extrapolated_lane = np.array(
        [x_intercept, img_height, x_intercept_at_min_y, min_y], np.float32)
    return extrapolated_lane


def draw_filtered_lines(img, img_height, img_width, lines, color=[255, 0, 0], thickness=8):
    # """
    # Idea:
        # find slope for each line and group the lines based on their slopes (slopes in certain range fall in the group)
        # extrapolate the grouped lines
    # """

    # Grouping lines
    left_lines, right_lines = separate_lines(lines, img_height, img_width)

    # Filter to remove the lines that doesn't constitute the lanes and construct single left and right lines
    left_filtered_lines = filter_lines(left_lines)
    right_filtered_lines = filter_lines(right_lines)

    # Extraolating the left and right lines until the border of the image (until y = 540-> image_height)
    if(len(left_filtered_lines) > 0):
        extrapolated_left_lane = extrapolate_lane(
            left_filtered_lines, img_height)
    else:
        extrapolated_left_lane = []

    if(len(right_filtered_lines) > 0):
        extrapolated_right_lane = extrapolate_lane(
            right_filtered_lines, img_height)
    else:
        extrapolated_right_lane = []

    extrapolated_lanes = []
    extrapolated_lanes.append(list(extrapolated_left_lane))
    extrapolated_lanes.append(list(extrapolated_right_lane))

    if(len(extrapolated_lanes) > 0):
        for line in extrapolated_lanes:
            if(len(line) == 4):
                cv2.line(img, (line[0], line[1]),
                         (line[2], line[3]), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_filtered_lines(line_img, img.shape[0], img.shape[1], lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)
