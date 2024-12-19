import collections
import cv2
import io
import numpy as np
import base64
import torch
from PIL import Image

class RegionConverter(object):
    def __init__(self, regions):
        self.regions = regions

    def encode(self, region_names):
        try:
            index = [self.regions.index(x) for x in region_names]
            return torch.LongTensor(index)
        except Exception as E:
            print(f"Error:{E}")
            return None


class StrLabelConverter(object):

    def __init__(self, alphabet, ignore_case=True):

        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-' + alphabet

        self.alphabet_indicies = {char: i for i, char in enumerate(self.alphabet)}

    def encode(self, text):
        try:
            if isinstance(text, str):
                # text = text.replace(',', '').replace('.', '')
                text = [self.alphabet_indicies[char.lower() if self._ignore_case else char] for char in text]
                length = [len(text)]
            elif isinstance(text, collections.Iterable):
                length = [len(s) for s in text]
                text = ''.join(text)
                text, _ = self.encode(text)
            return torch.LongTensor(text), torch.LongTensor(length)
        except Exception as E:
            print(text)
            print(f"Error: {E}")
            return None

    def decode(self, t, length, raw=False):

        if length.numel() == 1:
            length = length.item()
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i] == t[i - 1])):
                        char_list.append(self.alphabet[t[i]])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[:, i], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class DetectionUtils(object):
    def __init__(self, img_width, img_height, nms_thres, conf_thres):
        self.img_width = img_width
        self.img_height = img_height
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres

    def preprocess(self, img):
        img2 = img.copy()
        img2 = cv2.resize(img2, (self.img_width, self.img_height))
        img2 = img2.transpose((2, 0, 1))
        img2 = 2 * (img2 / 255 - 0.5)
        img2 = np.ascontiguousarray(img2.astype(np.float32))
        return img2

    def bbox_iou_np(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:

            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)

        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0,
                                                                                   None)

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou

    def nms_np(self, predictions, include_conf=True):
        filter_mask = (predictions[:, -1] >= self.conf_thres)
        predictions = predictions[filter_mask]

        if len(predictions) == 0:
            return np.array([])

        output = []

        while len(predictions) > 0:
            max_index = np.argmax(predictions[:, -1])

            if include_conf:
                output.append(predictions[max_index])
            else:
                output.append(predictions[max_index, :-1])

            ious = self.bbox_iou_np(np.array([predictions[max_index, :-1]]), predictions[:, :-1], x1y1x2y2=False)

            predictions = predictions[ious < self.nms_thres]

        return np.stack(output)

    def postprocess(self, predictions, origin_image, draw=True, squared=False):
        plates = self.nms_np(predictions)
        plate_images = []
        draw_image = origin_image.copy()
        flag = False
        if len(plates):
            flag = True
            plates[..., [4, 6, 8, 10]] += plates[..., [0]]
            plates[..., [5, 7, 9, 11]] += plates[..., [1]]
            ind = np.argsort(plates[..., -1])
            rx = float(origin_image.shape[1]) / self.img_width
            ry = float(origin_image.shape[0]) / self.img_height
            for index in ind:
                plate = plates[index]
                confidence_level = plate[-1]
                box = np.copy(plate[:12]).reshape(6, 2)
                box[:, ::2] *= rx
                box[:, 1::2] *= ry
                center_point = box[0]
                sizes = box[1]
                bbox = box[2:]
                correct_size_w = (bbox[2][0] - box[0][0]) / 2 + (bbox[3][0] - bbox[1][0]) / 2
                correct_size_h = (bbox[1][1] - bbox[0][1]) / 2 + (bbox[3][1] - bbox[2][1]) / 2
                correct_sizes = np.array([correct_size_w, correct_size_h], dtype=np.float32)

                coords = np.array(
                    [[0, 0], [0, correct_sizes[1]], [correct_sizes[0], 0], [correct_sizes[0], correct_sizes[1]]],
                    dtype='float32')

                transformation_matrix = cv2.getPerspectiveTransform(bbox, coords)

                lp_img = cv2.warpPerspective(origin_image, transformation_matrix, correct_sizes.astype(int))

                if squared:
                    ratio = abs((box[4, 0] - box[3, 0]) / (box[3, 1] - box[2, 1]))
                    if 2.6 >= ratio >= 0.8:
                        half = lp_img.shape[0] // 2
                        top = lp_img[:half, :]
                        bottom = lp_img[half:, :]
                        lp_img = hconcat_resize_min([top, bottom])

                if draw:
                    # for b in bbox:
                    #     cv2.circle(origin_image, b.astype(int), 8, (0, 0, 255), -1)
                    cv2.rectangle(draw_image, bbox[0].astype(int), bbox[-1].astype(int), (0, 255, 255), thickness=3, lineType=4)
                plate_images.append(lp_img)

        return draw_image, plate_images, flag


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for
                      im in im_list]
    return cv2.hconcat(im_list_resize)

def readb64(image_data):
    """
    Convert image data (bytes or base64 string) to numpy array
    """
    try:
        # If input is bytes, convert directly to numpy array
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Assume it's base64 string
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except Exception as e:
        raise ValueError(f"Error reading image: {str(e)}")


def image_to_base64(image: 'np.ndarray', image_format: str = 'jpg'):
    """
    Converts an OpenCV image to a Base64 string.

    """
    # Encode the image to the specified format
    success, buffer = cv2.imencode(f".{image_format}", image)
    if not success:
        raise ValueError("Could not encode the image to the base64 format.")

    # Convert the encoded image to Base64
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str


# def image_to_base64(image: np.ndarray) -> str:
#     pil_image = Image.fromarray(image)
#     buffered = io.BytesIO()
#     pil_image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")