from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, save_result_pyplot
import glob
import os
import time
import shutil
from PIL import Image
import traceback


def main():
    parser = ArgumentParser()
    parser.add_argument('--img_dirs', help='Image file')
    parser.add_argument('--output_dir')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for i, img_dir in enumerate(glob.glob(os.path.join(args.img_dirs, "*"))):
        os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'vis'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'crop'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, os.path.basename(img_dir), 'nodetected'), exist_ok=True)
        img_files = glob.glob(os.path.join(img_dir, "*"))
        # if i < 366:
        #     print("skip")
        #     continue
        for j, img in enumerate(img_files):
            #     if i == 366 and j < 140:
            #         print("skip")
            #         continue
            print(i, j, len(img_files), os.path.basename(img), os.path.basename(os.path.dirname(img)))
            output_path = os.path.join(args.output_dir, os.path.basename(img_dir), 'vis',
                                       os.path.splitext(os.path.basename(img))[0] + ".jpg")
            crop_output_path = os.path.join(args.output_dir, os.path.basename(img_dir), 'crop',
                                            os.path.splitext(os.path.basename(img))[0] + "_*.jpg")
            if os.path.isfile(output_path) and len(glob.glob(crop_output_path)) > 0:
                print("skip")
                continue
            # test a single image
            start = time.time()
            try:
                result = inference_detector(model, img)
            except:
                traceback.print_exc()
                continue
            print(time.time() - start)
            # show the results
            if len(result) < 1 or len(result[0]) < 1:
                shutil.copy(img, os.path.join(args.output_dir, 'nodetected'))
                continue

            save_result_pyplot(model, img, result, output_path, score_thr=args.score_thr)
            try:
                im = Image.open(img).convert("RGB")
            except:
                traceback.print_exc()
                continue
            for j, bbox in enumerate(result[0]):
                if bbox[4] < args.score_thr:
                    continue
                crop_im = im.crop([int(b) for b in bbox[:-1]])
                crop_im.save(
                    os.path.join(args.output_dir, os.path.basename(img_dir), 'crop',
                                 os.path.splitext(os.path.basename(img))[0] + "_{}.jpg".format(j)))


if __name__ == '__main__':
    main()
