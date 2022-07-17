import numpy as np
import cv2
from skimage import io


def segm_preprocess(self, path):
    name = path.split("/")[-1].split(".")[0]
    ext_path = path.split("/")[-3]
    retrieve_path = "ORIGINAL_PATH" + ext_path + "/image/" + name +".jpg"
    target_size = (512, 512)
    target_radius = 500

    img = cv2.imread(retrieve_path)
    img = self.scale_radius(img, target_radius)
    new_height, new_width = img.shape[0], img.shape[1]

    preproc = cv2.addWeighted(
        img, 4, cv2.GaussianBlur(img, (0, 0), target_radius / 30), -4, 128)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.20, param1=50, param2=30,
                               minRadius=int(new_width / 4), maxRadius=int(new_width / 2), minDist=1000)
    if circles is not None:
        circles = np.around(circles).astype(np.uint16)
        center = (circles[0, 0, 0], circles[0, 0, 1])
        radius = circles[0, 0, 2]
        center_x, center_y = center[1], center[0]
        if center_x + radius < new_height and center_y + radius < new_width:
            cropped = preproc[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            mask = np.zeros(preproc.shape, np.uint8)
            cv2.circle(mask, (center_y, center_x), int(radius * 0.95), (1, 1, 1), -1, 8, 0)
            mask = mask[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            cropped = mask * cropped
            mask = mask[:, :, 0] * 255
        else:
            radius = int(radius)
            center_x = int(center_x)
            center_y = int(center_y)
            pad_width = max(radius - center_x, radius + center_x - new_height) + 1
            preproc = np.stack([np.pad(preproc[:, :, c], pad_width)
                                for c in range(3)], axis=2)
            center_x += pad_width + 1
            center_y += pad_width + 1
            cropped = preproc[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            mask = np.zeros(preproc.shape, np.uint8)
            cv2.circle(mask, (center_y, center_x), int(
                radius * 0.95), (1, 1, 1), -1, 8, 0)
            mask = mask[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            tmp = radius + center_x - new_height
            mask[:pad_width, :pad_width - tmp, :] = 0
            mask[-pad_width:, -pad_width + tmp:, :] = 0
            cropped = mask * cropped
            mask = mask[:, :, 0] * 255
        # Resize to target size
        final = cv2.resize(cropped, target_size)
        mask = cv2.resize(mask, target_size)
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        if final.shape != (target_size[0], target_size[1], 3) or mask.shape != target_size:
            print("ERROR - Something went wrong with ", name)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        final = np.concatenate((final, mask.reshape(
            target_size[0], target_size[1], 1)), axis=-1)
        io.imsave(path, final)
    else:
        print("No fundus detected")
        print(retrieve_path)

def class_preprocess(self, path):
    name = path.split("/")[-1].split(".")[0]
    ext_path = path.split("/")[-2]
    retrieve_path = "ORIGINAL_PATH" + ext_path + "/" + name + ".jpg"
    target_size = (512, 512)
    target_radius = 500

    img = cv2.imread(retrieve_path)
    img = self.scale_radius(img, target_radius)
    new_height, new_width = img.shape[0], img.shape[1]

    preproc = cv2.addWeighted(
        img, 4, cv2.GaussianBlur(img, (0, 0), target_radius / 30), -4, 128)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.20, param1=50, param2=30,
                               minRadius=int(new_width / 4), maxRadius=int(new_width / 2), minDist=1000)
    if circles is not None:
        circles = np.around(circles).astype(np.uint16)
        center = (circles[0, 0, 0], circles[0, 0, 1])
        radius = circles[0, 0, 2]
        center_x, center_y = center[1], center[0]
        if center_x + radius < new_height and center_y + radius < new_width:
            cropped = preproc[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            mask = np.zeros(preproc.shape, np.uint8)
            cv2.circle(mask, (center_y, center_x), int(radius * 0.95), (1, 1, 1), -1, 8, 0)
            mask = mask[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            cropped = mask * cropped
            mask = mask[:, :, 0] * 255
        else:
            radius = int(radius)
            center_x = int(center_x)
            center_y = int(center_y)
            pad_width = max(radius - center_x, radius + center_x - new_height) + 1
            preproc = np.stack([np.pad(preproc[:, :, c], pad_width)
                                for c in range(3)], axis=2)
            center_x += pad_width + 1
            center_y += pad_width + 1
            cropped = preproc[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            mask = np.zeros(preproc.shape, np.uint8)
            cv2.circle(mask, (center_y, center_x), int(
                radius * 0.95), (1, 1, 1), -1, 8, 0)
            mask = mask[center_x - radius:center_x + radius, center_y - radius:center_y + radius, :]
            tmp = radius + center_x - new_height
            mask[:pad_width, :pad_width - tmp, :] = 0
            mask[-pad_width:, -pad_width + tmp:, :] = 0
            cropped = mask * cropped
            mask = mask[:, :, 0] * 255
        # Resize to target size
        final = cv2.resize(cropped, target_size)
        mask = cv2.resize(mask, target_size)
        mask[mask >= 128] = 255
        mask[mask < 128] = 0
        if final.shape != (target_size[0], target_size[1], 3) or mask.shape != target_size:
            print("ERROR - Something went wrong with ", name)
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        final = np.concatenate((final, mask.reshape(
            target_size[0], target_size[1], 1)), axis=-1)
        io.imsave(path, final)
    else:
        print("No fundus detected")
        print(retrieve_path)


def scale_radius(self, image, scale):
    # Roughly estimate fundus radius
    x = image[int(image.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2

    # Resize image to target radius value
    s = scale * 1.0 / r
    resized = cv2.resize(image, (0, 0), fx=s, fy=s)
    return resized