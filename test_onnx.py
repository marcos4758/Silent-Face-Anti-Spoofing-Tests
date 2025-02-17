import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredictONNX
from src.generate_patches import CropImage

warnings.filterwarnings("ignore")

SAMPLE_IMAGE_PATH = "./images/sample/"


def test(image_name, model_path_27, model_path_40, cpu_only=False):
    model_27 = AntiSpoofPredictONNX(model_path_27, cpu_only)
    model_40 = AntiSpoofPredictONNX(model_path_40, cpu_only)
    models = [
        {"model": model_27, "scale": 2.7},
        {"model": model_40, "scale": 4.0},
    ]

    image_cropper = CropImage()
    image_bgr = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    image_bbox = model_27.get_bbox(image_bgr)
    print("Image bbox:", image_bbox)

    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_dict in models:
        param = {
            "org_img": image_bgr,
            "bbox": image_bbox,
            "scale": model_dict["scale"],
            "out_w": model_dict["model"].input_size[0],
            "out_h": model_dict["model"].input_size[1],
            "crop": True,
        }
        img_crop_bgr = image_cropper.crop(**param)
        start = time.time()
        prediction += model_dict["model"].get_prediction(img_crop_bgr)
        test_speed += time.time() - start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image_bgr,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color,
        2,
    )
    cv2.putText(
        image_bgr,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5 * image_bgr.shape[0] / 1024,
        color,
    )

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image_bgr)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--model_path_27", type=str, help="Path to .onnx model weights (MiniFASNetV2 2.7)")
    parser.add_argument("--model_path_40", type=str, help="Path to .onnx model weights (MiniFASNetV1SE 4.0)")
    parser.add_argument(
        "--cpu_only", action=argparse.BooleanOptionalAction, default=False, help="Whether to use CPU only"
    )
    parser.add_argument("--image_name", type=str, default="image_F1.jpg", help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_path_27, args.model_path_40, args.cpu_only)
