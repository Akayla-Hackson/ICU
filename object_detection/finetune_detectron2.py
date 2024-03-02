from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def main():
    train_data = './data/bball_train'
    val_data = './data/bball_val'
    train_gt = './data/annotations/bball_train.json'
    val_gt = './data/annotations/bball_val.json'

    # Registering the COCO datasets
    register_coco_instances("bball_train", {}, train_gt, train_data)
    register_coco_instances("bball_val", {}, val_gt, val_data)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Update the dataset names
    cfg.DATASETS.TRAIN = ("bball_train",)
    cfg.DATASETS.TEST = ("bball_val",)
    cfg.OUTPUT_DIR = './runs/detect/'

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 ]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

    # Now you can initialize the trainer and start training
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == '__main__':
    main()

