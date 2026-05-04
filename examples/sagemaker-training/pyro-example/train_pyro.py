import tlc
import tlc_ultralytics

project_name = "hf-pyronear"
dataset_name = "pyro-sdis"

train_table = tlc.Table.from_names(project_name=project_name, dataset_name=dataset_name, table_name="train")
val_table = tlc.Table.from_names(project_name=project_name, dataset_name=dataset_name, table_name="val")

settings = tlc_ultralytics.Settings(project_name=project_name)
model = tlc_ultralytics.YOLO("yolo11s.pt")

model.train(
    tables={"train": train_table, "val": val_table},
    settings=settings,
    epochs=10,
    batch=8,
    imgsz=640,
    device="mps",
    workers=4,
    verbose=True,
)