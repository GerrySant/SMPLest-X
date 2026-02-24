import os
import argparse
import json
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

from ultralytics import YOLO

from human_models.human_models import SMPLX
from main.base import Tester
from main.config import Config
from utils.data_utils import process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression


# --------------------------------------------------
# Utils
# --------------------------------------------------

def extract_smplx_keypoints(out, cfg):

    joints_2d = out["smplx_joint_proj"][0].cpu().numpy()
    joints_3d = out["smplx_joint_cam"][0].cpu().numpy()
    bb2img_trans = out["bb2img_trans"][0].cpu().numpy()

    # ---- Upscale from heatmap to body resolution ----
    joints_2d[:, 0] = (joints_2d[:, 0]) / cfg.model.output_hm_shape[2] * cfg.model.input_body_shape[1]
    joints_2d[:, 1] = (joints_2d[:, 1] + 0.5) / cfg.model.output_hm_shape[1] * cfg.model.input_body_shape[0]

    # homogeneous coords
    ones = np.ones((joints_2d.shape[0], 1))
    joints_homo = np.concatenate([joints_2d, ones], axis=1)

    # map back to original image
    joints_2d_img = (bb2img_trans @ joints_homo.T).T
    return joints_2d_img.tolist(), joints_3d.tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to a .mp4 file or to a directory containing .mp4 files")
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--json_output_path", type=str, required=True,
                        help="Output directory (if video_path is a directory) or JSON file path (if single video)")

    parser.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--multi_person", action="store_true")

    return parser.parse_args()

# --------------------------------------------------
# Core video processing
# --------------------------------------------------

def process_single_video(
    video_path: Path,
    json_out_path: Path,
    cfg,
    tester,
    detector,
    smpl_x,
    stride: int,
    max_frames: int,
    multi_person: bool,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_frames = min(num_frames, max_frames) if max_frames else num_frames

    transform = transforms.ToTensor()

    all_data = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "fps": fps,
        "image_size": [width, height],
        "joint_type": "smplx",
        "num_joints": len(smpl_x.joints_name),
        "joint_names": list(smpl_x.joints_name),
        "frames": [],
    }

    frame_idx = 0
    processed_idx = 0

    for _ in tqdm(
        range(total_frames),
        desc=f"Processing {video_path.name}",
        leave=False,
    ):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        # BGR -> RGB -> float32 (equivalente a load_img)
        original_img = frame_bgr[:, :, ::-1].astype(np.float32)
        vis_img = original_img.copy()
        h, w = original_img.shape[:2]

        frame_entry = {
            "frame_id": frame_idx,
            "image_size": [width, height],
            "timestamp_sec": frame_idx / fps,
            "persons": [],
        }

        # YOLO detection
        yolo_out = detector.predict(
            original_img,
            device="cuda",
            classes=0,
            conf=cfg.inference.detection.conf,
            verbose=False,
        )[0]

        bboxes = yolo_out.boxes.xyxy.detach().cpu().numpy()
        if len(bboxes) == 0:
            all_data["frames"].append(frame_entry)
            continue

        if not multi_person:
            bboxes = bboxes[:1]
        else:
            bboxes = non_max_suppression(bboxes, cfg.inference.detection.iou_thr)

        for pid, bb in enumerate(bboxes):
            bbox_xywh = np.array(
                [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]], dtype=np.float32
            )

            bbox = process_bbox(
                bbox_xywh,
                img_width=w,
                img_height=h,
                input_img_shape=cfg.model.input_img_shape,
                ratio=getattr(cfg.data, "bbox_ratio", 1),
            )

            img_patch, img2bb_trans, bb2img_trans = generate_patch_image(
                original_img,
                bbox=bbox,
                scale=1.0,
                rot=0.0,
                do_flip=False,
                out_shape=cfg.model.input_body_shape,
                #out_shape=cfg.model.input_img_shape,
            )

            img_tensor = transform(img_patch) / 255.0
            img_tensor = img_tensor.cuda()[None]

            # with torch.no_grad():
            #     out = tester.model({"img": img_tensor}, {}, {}, "test")
            #out["bb2img_trans"]=bb2img_trans
            with torch.no_grad():
                out = tester.model(
                    {"img": img_tensor},
                    {},
                    {"bb2img_trans": torch.from_numpy(bb2img_trans)[None].float().cuda()},
                    "test",
                )
            
            joints_2d, joints_3d = extract_smplx_keypoints(out, cfg)
            cam_trans = out["cam_trans"][0].detach().cpu().numpy().tolist()

            frame_entry["persons"].append(
                {
                    "person_id": int(pid),
                    "bbox": [float(v) for v in bbox],
                    "cam_trans": cam_trans,
                    "joints_2d": joints_2d,
                    "joints_3d": joints_3d,
                }
            )

        all_data["frames"].append(frame_entry)
        processed_idx += 1

    cap.release()

    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Saved: {json_out_path}  |  Frames: {processed_idx}/{frame_idx}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    args = parse_args()
    cudnn.benchmark = True

    root_dir = Path(__file__).resolve().parent.parent
    video_path = Path(args.video_path)

    config_path = root_dir / "pretrained_models" / args.ckpt_name / "config_base.py"
    checkpoint_path = root_dir / "pretrained_models" / args.ckpt_name / f"{args.ckpt_name}.pth.tar"

    cfg = Config.load_config(str(config_path))
    cfg.update_config(
        {
            "model": {"pretrained_model_path": str(checkpoint_path)},
            "log": {
                "exp_name": f"inference_video_batch_{datetime.datetime.now():%Y%m%d_%H%M%S}",
                "log_dir": str(root_dir / "outputs"),
            },
        }
    )
    cfg.prepare_log()

    smpl_x = SMPLX(cfg.model.human_model_path)
    tester = Tester(cfg)
    tester._make_model()

    detector = YOLO(
        getattr(cfg.inference.detection, "model_path", "./pretrained_models/yolov8x.pt")
    )

    if video_path.is_file():
        if video_path.suffix.lower() != ".mp4":
            raise ValueError(f"Expected .mp4 file, got: {video_path}")

        json_out = Path(args.json_output_path)
        if json_out.suffix != ".json":
            raise ValueError("For single video, json_output_path must be a .json file")

        videos = [video_path]
        json_paths = [json_out]

    elif video_path.is_dir():
        videos_all = sorted(video_path.glob("*.mp4"))
        if len(videos_all) == 0:
            raise RuntimeError(f"No .mp4 files found in directory: {video_path}")

        json_base = Path(args.json_output_path)
        json_base.mkdir(parents=True, exist_ok=True)

        videos = []
        json_paths = []

        for vid in videos_all:
            json_out = json_base / f"{vid.stem}.json"
            if not json_out.exists():
                videos.append(vid)
                json_paths.append(json_out)

        if len(videos) == 0:
            print("✅ All videos already processed. Nothing to do.")
            return

        print(f"▶️ Processing {len(videos)} / {len(videos_all)} videos (skipping existing JSONs)")

    else:
        raise RuntimeError(f"Invalid video_path: {video_path}")

    for vid, json_out in zip(videos, json_paths):
        process_single_video(
            video_path=vid,
            json_out_path=json_out,
            cfg=cfg,
            tester=tester,
            detector=detector,
            smpl_x=smpl_x,
            stride=args.stride,
            max_frames=args.max_frames,
            multi_person=args.multi_person,
        )

if __name__ == "__main__":
    main()
