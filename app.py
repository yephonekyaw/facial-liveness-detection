import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from src.preprocessing.face_pipeline import FaceDetector, crop_and_resize, expand_to_square
from src.training.eval_runner import load_model
from src.data.transforms import eval_transform
from src.config import CHECKPOINT_DIR

DEFAULT_CKPT = CHECKPOINT_DIR / "both" / "best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
detector = None
transform = None


def load_globals(ckpt_path: Path):
    global model, detector, transform
    model, _ = load_model(ckpt_path, device)
    detector = FaceDetector(det_size=(640, 640))
    transform = eval_transform()


def predict(image_rgb: np.ndarray):
    if image_rgb is None:
        return None, "No image", 0.0

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    box = detector.detect(bgr)

    if box is None:
        return image_rgb, "No face detected", 0.0

    box_sq = expand_to_square(box, bgr.shape)
    crop_bgr = crop_and_resize(bgr, box_sq, size=256)

    transformed = transform(image=crop_bgr)
    x = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit.float()).item()  # >0.5 → attack

    color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    cv2.rectangle(crop_rgb, (0, 0), (255, 255), color, 6)

    label = f"{'ATTACK' if prob > 0.5 else 'REAL'}  —  score: {prob:.3f}"
    return crop_rgb, label, round(prob, 4)


def build_ui():
    with gr.Blocks(title="AttackNet Live Demo") as demo:
        gr.Markdown("## AttackNet v2.2 — Facial Liveness Detection")
        with gr.Row():
            with gr.Column():
                webcam = gr.Image(
                    sources=["webcam"], streaming=True, type="numpy", label="Webcam"
                )
                btn = gr.Button("Run once (snapshot)", variant="secondary")
            with gr.Column():
                crop_out = gr.Image(label="Face crop (256×256)")
                verdict = gr.Textbox(label="Verdict")
                score = gr.Number(label="Attack probability  (0 = real, 1 = attack)")

        webcam.stream(predict, inputs=webcam, outputs=[crop_out, verdict, score], stream_every=3)
        btn.click(predict, inputs=webcam, outputs=[crop_out, verdict, score])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    load_globals(args.checkpoint)
    build_ui().launch(server_name='0.0.0.0', server_port=7860, share=False)
