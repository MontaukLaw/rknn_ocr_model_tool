import numpy as np
import cv2
from rknn.api import RKNN
import torch
# from label_convert import CTCLabelConverter

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

class DBPostProcess():
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=2):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, h_w_list, is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        h_w_list: 包含[h,w]的数组
        pred:
            binary: text region segmentation map, with shape (N, 1,H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = self.binarize(pred)
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = h_w_list[batch_index]
            boxes, scores = self.post_p(pred[batch_index], segmentation[batch_index], width, height,
                                        is_output_polygon=is_output_polygon)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def binarize(self, pred):
        return pred > self.thresh

    def post_p(self, pred, bitmap, dest_width, dest_height, is_output_polygon=False):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''
        height, width = pred.shape
        boxes = []
        new_scores = []
        # bitmap = bitmap.cpu().numpy()
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            four_point_box, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            if not is_output_polygon:
                box = np.array(four_point_box)
            else:
                box = box.reshape(-1, 2)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            new_scores.append(score)
        return boxes, new_scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        # bitmap = bitmap.detach().cpu().numpy()
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def narrow_224_32(image, expected_size=(224,32)):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    # scale = eh / ih
    scale = min((eh/ih),(ew/iw))
    # scale = eh / max(iw,ih)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

    top = 0
    bottom = eh - nh
    left = 0
    right = ew - nw

    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image,new_img

def draw_bbox(img_path, result, color=(0, 0, 255), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path

if __name__ == '__main__':

    post_proess = DBPostProcess()
    is_output_polygon = False
    # Create RKNN object
    rknn = RKNN()

    ret = rknn.load_rknn('static_det.rknn')

    # Set inputs
    img = cv2.imread('./test.jpg')
    origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img0 ,image = narrow_224_32(img,expected_size=(640,640))

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rv1126',device_id="a9d00ab1f032c17a")
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[image])

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[image])
    print('done')

    feat_2 = torch.from_numpy(outputs[0])
    print(feat_2.size())
    box_list, score_list = post_proess(outputs[0], [image.shape[:2]], is_output_polygon=is_output_polygon)
    box_list, score_list = box_list[0], score_list[0]
    if len(box_list) > 0:
        idx = [x.sum() > 0 for x in box_list]
        box_list = [box_list[i] for i, v in enumerate(idx) if v]
        score_list = [score_list[i] for i, v in enumerate(idx) if v]
    else:
        box_list, score_list = [], []

    img = draw_bbox(image, box_list)
    img = img[0:img0.shape[0],0:img0.shape[1]]
    cv2.imshow("img",img)
    cv2.waitKey()
    rknn.release()
