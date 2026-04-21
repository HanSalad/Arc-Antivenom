from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=64,
        device="0",
        workers=6,
        project="runs",
        name="game_targets",
        
    )

    print(results)


if __name__ == "__main__":
    main()