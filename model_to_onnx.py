import torch
import onnx
import onnxsim

from src.anti_spoof_predict import AntiSpoofPredict
from src.utility import parse_model_name

import os

import argparse


def check_onnx_model(model):
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("ONNX model is invalid:", e)
    else:
        print("ONNX model is valid!")


if __name__ == "__main__":
    # parsing arguments
    p = argparse.ArgumentParser(description="Convert model weights from .pth to .onnx")
    p.add_argument("--model_path_27", type=str, help="Path to .pth model weights (MiniFASNetV2 2.7)")
    p.add_argument("--model_path_40", type=str, help="Path to .pth model weights (MiniFASNetV1SE 4.0)")
    p.add_argument(
        "--print_summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to print the model information (torchinfo is needed)",
    )
    args = p.parse_args()

    assert os.path.isfile(args.model_path_27), "Model {} not found!".format(args.model_path_27)
    assert os.path.isfile(args.model_path_40), "Model {} not found!".format(args.model_path_40)

    model_27 = AntiSpoofPredict(0)
    model_27._load_model(args.model_path_27)
    model_27.model.eval()
    h_input, w_input, _, _ = parse_model_name(os.path.basename(args.model_path_27))
    model_27_input_size = (h_input, w_input)

    model_40 = AntiSpoofPredict(0)
    model_40._load_model(args.model_path_40)
    model_40.model.eval()
    h_input, w_input, _, _ = parse_model_name(os.path.basename(args.model_path_40))
    model_40_input_size = (h_input, w_input)

    models = [
        {"model": model_27.model, "path": args.model_path_27, "input_size": model_27_input_size},
        {"model": model_40.model, "path": args.model_path_40, "input_size": model_40_input_size},
    ]

    print("Models loaded successfully")

    if args.print_summary:
        from torchinfo import summary

        for model_dict in models:
            print("\nModel summary:", model_dict["path"])
            summary(model_dict["model"], input_size=(1, 3, model_dict["input_size"][0], model_dict["input_size"][1]))

    # Convert model to onnx
    for model_dict in models:
        model = model_dict["model"]
        path = model_dict["path"]
        input_size = model_dict["input_size"]
        onnx_path = path.replace(".pth", ".onnx")

        # Save onnx model
        model.eval()
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(next(model.parameters()).device)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            # verbose=False,
            input_names=["input"],
            output_names=["output"],
            export_params=True,
        )

        # Load onnx model
        onnx_model = onnx.load(onnx_path)

        # Check onnx model
        print("\nCheck exported model ", onnx_path)
        check_onnx_model(onnx_model)

        # Simplify the model
        print("\nSimplify model for", onnx_path)
        onnx_model, check = onnxsim.simplify(onnx_model, check_n=3)
        assert check, f"Simplified ONNX model ({onnx_path}) could not be validated"

        # Check simplified model
        print("\nCheck simplified model for", onnx_path)
        check_onnx_model(onnx_model)

        # Save simplified model
        onnx.save(onnx_model, onnx_path)

        print("\nIR version:", onnx_model.ir_version)
        print("\nONNX model exported to:", onnx_path)
