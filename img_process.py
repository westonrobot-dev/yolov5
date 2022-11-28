from thermal.utils import calc_temps_in_bounding_box, drawBoundingBoxes
import cv2

test_img_file = "data/thermal_images/thermal_1.png"
test_img_files = [f"data/thermal_images/thermal_{i}.png" for i in range(1, 500)]
test_bounding_box = [250, 290, 260, 300]   # in a format of bb_x_min(top), bb_x_max(bottom), bb_y_min(left), bb_y_max(right)


if __name__ == '__main__':
    image = cv2.imread(test_img_file, 0)

    # total_read = 0
    # for file in test_img_files:
    #     print(file.split(".")[0].split("_")[-1])
    #     res = calc_temps_in_bounding_box(file, test_bounding_box)
    #     if res: total_read += 1
    #     print(res)
    # print(total_read/len(test_img_files))

    output = calc_temps_in_bounding_box(test_img_file, test_bounding_box)
    print(output)
    imageData = drawBoundingBoxes(imageData=image, bounding_box=test_bounding_box, label=f"average temp: {output[2]} ")
    cv2.imshow('average temperature within the bounding box', imageData)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    