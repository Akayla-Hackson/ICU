from ultralytics import YOLO
import torch
import argparse
from torch.utils import tensorboard

def main(args):
    # Load the pretrained YOLOv8 model
    model = YOLO('yolov8n.pt') 

    results = model.train(data='./data/train_val_bball.yaml', epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, plots=True, lr0=args.init_learning_rate, lrf=args.final_learning_rate, optimizer=args.optimizer, freeze=args.freeze)

    # Save the trained model weights
    trained_weights_path = f'./finetuned_yolov8n_weights_E{args.epochs}_B{args.batch}InitLr{args.init_learning_rate}_FinLr{args.final_learning_rate}_Opt{args.optimizer}_imgSz{args.imgsz}_freeze{args.freeze}.pt'
    torch.save(model.state_dict(), trained_weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Num of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--init_learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--final_learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='auto', help='optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]')
    #parser.add_argument('--freeze', type=int, default=None, help='(int or list, optional) freeze first n layers, or freeze list of layer indices during training')
    parser.add_argument('--freeze', nargs='+', default=None, help='(int or list, optional) freeze first n layers, or freeze list of layer indices during training')
    args = parser.parse_args()
    main(args)