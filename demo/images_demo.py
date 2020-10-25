from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, save_result_pyplot
import glob
import os
import time


def main():
    parser = ArgumentParser()
    parser.add_argument('--imgs', help='Image file')
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

    os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'crop'), exist_ok=True)
    img_files = glob.glob(args.imgs)
    for img in img_files:
        print(os.path.basename(img))
        # test a single image
        start = time.time()
        result = inference_detector(model, img)
        print(time.time() - start)
        # show the results
        print(result)
        sys.exit()
        output_path = os.path.join(args.output_dir, 'vis', os.path.splitext(os.path.basename(img))[0] + ".jpg")
        save_result_pyplot(model, img, result, output_path, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
