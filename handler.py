"""
KayTrain Worker — RunPod Serverless Handler
Handles two actions:
  - "test"  : Verify GPU environment and return system info
  - "train" : Download dataset, train R(2+1)D-18 model, export ONNX
"""

import runpod
import os
import json
import time
import torch
import torchvision


# ─────────────────────────────────────────────
#  ACTION: test
# ─────────────────────────────────────────────
def action_test(job_input: dict) -> dict:
    """Verify the GPU worker is alive and CUDA is working."""
    print("[KayTrain] Running system test...")

    result = {
        "status": "ok",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "vram_gb": None,
        "r2plus1d_ok": False,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result["gpu_name"] = props.name
        result["vram_gb"] = round(props.total_memory / 1024 ** 3, 1)

        # Quick inference test
        model = torchvision.models.video.r2plus1d_18(weights=None)
        model.eval().cuda()
        dummy = torch.randn(1, 3, 8, 112, 112).cuda()
        with torch.no_grad():
            out = model(dummy)
        result["r2plus1d_ok"] = out.shape == torch.Size([1, 400])
        del model, dummy
        torch.cuda.empty_cache()

    print(f"[KayTrain] Test result: {json.dumps(result, indent=2)}")
    return result


# ─────────────────────────────────────────────
#  ACTION: train
# ─────────────────────────────────────────────
def action_train(job_input: dict) -> dict:
    """
    Full training pipeline:
    1. Download dataset ZIP from Google Drive URL
    2. Fine-tune R(2+1)D-18 for 18-label multi-label classification
    3. Export best checkpoint as ONNX
    4. Upload result and return download URL
    """
    import zipfile
    import urllib.request
    import shutil

    dataset_url = job_input.get("dataset_url")
    config = job_input.get("config", {})
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 8)
    lr = config.get("learning_rate", 3e-4)
    patience = config.get("early_stopping_patience", 7)
    num_labels = config.get("num_labels", 18)
    model_version = config.get("model_version", "v1")

    if not dataset_url:
        return {"status": "error", "message": "dataset_url is required"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[KayTrain] Device: {device}")
    print(f"[KayTrain] Config: epochs={epochs}, batch={batch_size}, lr={lr}, labels={num_labels}")

    # ── 1. Download dataset ──────────────────────────────
    os.makedirs("/workspace/data", exist_ok=True)
    zip_path = "/workspace/data/dataset.zip"
    print(f"[KayTrain] Downloading dataset from S3 Network Volume ...")

    # Due to RunPod's Cloudflare Proxy blocking Presigned URLs, we download using passed AWS credentials
    s3_cfg = job_input.get("s3_cfg", {})
    if s3_cfg:
        import boto3, re
        region_match = re.search(r's3api-(.*?)\.runpod\.io', s3_cfg["endpoint"])
        region = region_match.group(1) if region_match else "us-east-1"
        s3 = boto3.client(
            's3',
            endpoint_url=s3_cfg["endpoint"],
            aws_access_key_id=s3_cfg["access_key"],
            aws_secret_access_key=s3_cfg["secret_key"],
            region_name=region
        )
        s3.download_file(s3_cfg["bucket"], "dataset.zip", zip_path)
    else:
        req = urllib.request.Request(dataset_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
    print(f"[KayTrain] Download complete: {os.path.getsize(zip_path) / 1024**2:.1f} MB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/workspace/data/")
    print("[KayTrain] Dataset extracted.")

    # ── 2. Build DataLoader ──────────────────────────────
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

    class FlowDataset(Dataset):
        """
        Expects dataset structure:
          /workspace/data/flows/   ← .npy optical flow tensors, shape (16, 224, 224, 2)
          /workspace/data/labels.json ← {"filename.npy": [0,1,0,...,1]}
        """
        def __init__(self, flows_dir, labels_file):
            with open(labels_file) as f:
                self.labels = json.load(f)
            self.files = [
                os.path.join(flows_dir, k)
                for k in self.labels.keys()
                if os.path.exists(os.path.join(flows_dir, k))
            ]
            print(f"[KayTrain] Dataset: {len(self.files)} samples")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            path = self.files[idx]
            key = os.path.basename(path)
            flow = np.load(path).astype(np.float32)  # (16, H, W, 2)
            # Normalize and reshape to (3, 16, 112, 112) for R(2+1)D
            # Use dx, dy as 2 channels, magnitude as 3rd channel
            flow_resized = []
            import cv2
            for t in range(flow.shape[0]):
                f = flow[t]  # (H, W, 2)
                dx = cv2.resize(f[:, :, 0], (112, 112))
                dy = cv2.resize(f[:, :, 1], (112, 112))
                mag = np.sqrt(dx**2 + dy**2)
                flow_resized.append(np.stack([dx, dy, mag], axis=0))  # (3, 112, 112)
            tensor = np.stack(flow_resized, axis=1)  # (3, 16, 112, 112)
            # Normalize to [-1, 1]
            tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
            label = np.array(self.labels[key], dtype=np.float32)
            return torch.from_numpy(tensor), torch.from_numpy(label)

    flows_dir = "/workspace/data/flows"
    labels_file = "/workspace/data/labels.json"

    dataset = FlowDataset(flows_dir, labels_file)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    def variable_length_collate(batch):
        # batch is a list of tuples: (tensor_3_T_H_W, label)
        from torch.nn.utils.rnn import pad_sequence
        
        tensors = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # tensor shape is (3, T, 112, 112)
        # pad_sequence expects (T, *). So we permute (3, T, 112, 112) -> (T, 3, 112, 112)
        permuted = [t.permute(1, 0, 2, 3) for t in tensors]
        padded = pad_sequence(permuted, batch_first=True, padding_value=0.0)
        # padded shape is (Batch, Max_T, 3, 112, 112)
        
        # permute back to (Batch, 3, Max_T, 112, 112)
        final_tensors = padded.permute(0, 2, 1, 3, 4)
        final_labels = torch.stack(labels)
        return final_tensors, final_labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=variable_length_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=variable_length_collate)

    # ── 3. Build Model ───────────────────────────────────
    print("[KayTrain] Loading R(2+1)D-18 pretrained...")
    model = torchvision.models.video.r2plus1d_18(weights="DEFAULT")
    # Replace final layer: 400 classes → num_labels
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 4. Training Loop ─────────────────────────────────
    def compute_map(preds, targets, threshold=0.5):
        from sklearn.metrics import average_precision_score
        preds_np = torch.sigmoid(preds).cpu().numpy()
        targets_np = targets.cpu().numpy()
        try:
            return average_precision_score(targets_np, preds_np, average="macro")
        except Exception:
            return 0.0

    best_map = 0.0
    best_checkpoint = "/workspace/best_model.pt"
    no_improve = 0
    history = []

    print(f"[KayTrain] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item()
                all_preds.append(out)
                all_targets.append(y)
        val_loss /= len(val_loader)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        current_map = compute_map(all_preds, all_targets)

        scheduler.step()
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "val_loss": round(val_loss, 4), "mAP": round(current_map * 100, 2)}
        history.append(entry)
        print(f"[KayTrain] Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} "
              f"| Val: {val_loss:.4f} | mAP: {current_map*100:.1f}%")

        if current_map > best_map:
            best_map = current_map
            no_improve = 0
            torch.save(model.state_dict(), best_checkpoint)
            print(f"[KayTrain] ✅ New best mAP: {best_map*100:.1f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[KayTrain] Early stopping at epoch {epoch}")
                break

    # ── 5. Export ONNX ───────────────────────────────────
    print("[KayTrain] Exporting to ONNX...")
    model.load_state_dict(torch.load(best_checkpoint))
    model.eval().cpu()

    onnx_path = f"/workspace/kayai_motion_{model_version}.onnx"
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["flow_input"],
        output_names=["logits"],
        dynamic_axes={"flow_input": {0: "batch_size"}},
        opset_version=17
    )
    size_mb = os.path.getsize(onnx_path) / 1024 ** 2
    print(f"[KayTrain] ONNX exported: {onnx_path} ({size_mb:.1f} MB)")

    # ── 6. Upload ONNX back to S3 ────────────────────────
    download_url = None
    if s3_cfg:
        print("[KayTrain] Uploading ONNX to S3...")
        import boto3, re
        region_match = re.search(r's3api-(.*?)\.runpod\.io', s3_cfg["endpoint"])
        region = region_match.group(1) if region_match else "us-east-1"
        s3_upload = boto3.client(
            's3',
            endpoint_url=s3_cfg["endpoint"],
            aws_access_key_id=s3_cfg["access_key"],
            aws_secret_access_key=s3_cfg["secret_key"],
            region_name=region
        )
        onnx_key = f"models/kayai_motion_{model_version}.onnx"
        s3_upload.upload_file(onnx_path, s3_cfg["bucket"], onnx_key)
        print(f"[KayTrain] ONNX uploaded to s3://{s3_cfg['bucket']}/{onnx_key}")
        download_url = f"s3://{s3_cfg['bucket']}/{onnx_key}"
    else:
        print("[KayTrain] WARNING: No s3_cfg provided, cannot upload ONNX.")

    return {
        "status": "completed",
        "model_version": model_version,
        "best_mAP": round(best_map * 100, 2),
        "epochs_trained": len(history),
        "model_size_mb": round(size_mb, 1),
        "download_url": download_url,
        "onnx_key": onnx_key if s3_cfg else None,
        "history": history[-5:],  # Last 5 epochs
    }


# ─────────────────────────────────────────────
#  MAIN HANDLER
# ─────────────────────────────────────────────
def handler(event):
    job_input = event.get("input", {})
    action = job_input.get("action", "test")

    print(f"[KayTrain] Handler called | action={action}")

    if action == "test":
        return action_test(job_input)
    elif action == "train":
        return action_train(job_input)
    else:
        return {"status": "error", "message": f"Unknown action: {action}. Use 'test' or 'train'."}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
