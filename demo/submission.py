import json
import numpy as np
from pycocotools import mask as mutils
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import os
import cv2


def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


# Used only for testing.
# This is copied from https://www.kaggle.com/paulorzp/run-length-encode-and-decode.
# Thanks to Paulo Pinto.
def rle_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def mask_to_poly(mask):
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)
    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
    #                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
    if len(segmentation) == 0:
        return None
    return segmentation[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_file', help='Image file')
    parser.add_argument('--output_dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filepath = args.result_file
    with open(filepath) as json_file:
        json_data = json.load(json_file)

    encoded_pixels = []
    img_ids = []
    height = []
    width = []
    category_ids = []

    for i in tqdm(range(len(json_data))):
        mask = mutils.decode(json_data[i]['segmentation'])
        poly = mask_to_poly(mask)
        print(poly)
        sys.exit()
        encoded_pixels.append(rle_to_string(rle_encode(mutils.decode(json_data[i]['segmentation']))))
        img_ids.append(json_data[i]['image_id'])
        category_ids.append(json_data[i]['category_id'])
        height.append(json_data[i]['segmentation']['size'][0])
        width.append(json_data[i]['segmentation']['size'][1])
    data = {'ImageId': img_ids,
            'EncodedPixels': encoded_pixels,
            'Height': height,
            'Width': width,
            'CategoryId': category_ids}
    submission = pd.DataFrame(data)
    answer_dummy = submission.sample(50)

    submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=False)
    answer_dummy.to_csv(os.path.join(args.output_dir, 'answer_dummy.csv'), index=False)


if __name__ == '__main__':
    main()
