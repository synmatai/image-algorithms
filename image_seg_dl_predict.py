import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import numpy as np
import albumentations as A
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import argparse
from PIL import Image
import glob
from skimage import io, color
import tifffile as tif
from skimage import io, measure, color 
from albumentations import Compose, LongestMaxSize, PadIfNeeded
import torch.nn.functional as F
from torchvision import transforms as torchtrans
import copy
from collections import Counter
import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#图像分割模型
class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs): 
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)

        self.number_of_classes = out_classes
        if in_channels == 1:  # 如果是单通道图像，使用第一个通道的均值和标准差
            mean = params["mean"][0]
            std = params["std"][0]
            self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 1, 1, 1))
        else:
            # 对于多通道图像的情况
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = image.to(self.device)
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class SingleImageDataset(BaseDataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = io.imread(self.image_path)
        if image.shape[2] == 4:  # 检查是否是 RGBA 图像
            image = image[:, :, :3]  # 去掉 Alpha 通道
        mask = np.array(cv2.imread(self.image_path.replace("Images", "GT"), 0))
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = torchtrans.ToTensor()(image)
        return image, mask


def closest_multiple_of_32(size):
    return (size + 31) // 32 * 32

def get_transform(image_size, output_stride=32):
    height, width = image_size
    new_height = closest_multiple_of_32(height)
    new_width = closest_multiple_of_32(width)
    return Compose([
        LongestMaxSize(max_size=max(new_height, new_width)),
        PadIfNeeded(min_height=new_height, min_width=new_width, always_apply=True)
    ], p=1)

#训练验证集需要mask
class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = np.array(cv2.imread(self.masks_fps[i], 0))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        image = torchtrans.ToTensor()(image)
        # return image, mask_remap
        return image, mask

    def __len__(self):
        return len(self.ids)
    
def process_images_train_val(train_image_path,val_image_path):
    """
    读取文件夹下的所有图片，检查图片的长宽是否能被32整除。
    如果不能，则扩充到最近的32的倍数，并返回所有扩充后图片的最大长宽。

    :param folder_path: 文件夹路径
    :return: (max_width, max_height) 扩充后的最大长宽
    """
    max_width, max_height = 0,0
    files = glob.glob(train_image_path+'/*')
    files += glob.glob(val_image_path+'/*')
    for file_path in files:
        try:
            with Image.open(file_path) as img:
                height, width = img.size
                # 计算扩充后的尺寸
                new_width = (width + 31) // 32 * 32
                new_height = (height + 31) // 32 * 32
                
                max_width = max(max_width, new_width)
                max_height = max(max_height, new_height)
        except Exception as e:
            print(f"无法处理文件 {file_path}: {e}")

    return max_width, max_height

#数据增强统一尺寸
def get_training_augmentation(min_height=448,min_width=608):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=min_height, min_width=min_width, always_apply=True),
        A.RandomCrop(height=min_height, width=min_width, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

def tds(areas, areaSum, areaImage):
    """
    计算平均面积、标准差和孔隙率
    :param areas: 区域面积列表
    :param areaSum: 区域总面积
    :param areaImage: 图像总面积
    :return: 平均面积、标准差、孔隙率
    """
    if len(areas) == 0:
        return 0, 0, 0

    avg =  areaSum / len(areas)
    std = np.std(areas)
    porosity = areaSum / areaImage * 100

    return avg, std, porosity

def evaluate(pred, gt, numclasses=3):
    #px = 0.00476562
    px = 0.0046875
    colors = ['blue', 'green', 'gray']
    if numclasses == 2:
        contours, _= cv2.findContours(pred.squeeze(0).numpy().astype(np.uint8) , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for k in range(len(contours)):
            conmap= np.zeros((pred.shape[1],pred.shape[2]),dtype=np.uint8)
            cv2.drawContours(conmap, contours, k, 255, -1)
            area = cv2.countNonZero(conmap)
            if area>=5:
                areas.append(area)
        areas.sort()
        areaSum = sum(areas)
        count = Counter(areas)
        percent_dict= {}
        for size, cnt in count.items():
            percent=(size*cnt/areaSum)*100
            percent_dict[size*px] = percent
        x_set = list(percent_dict.keys())
        y_set = list(percent_dict.values())
        y_set = list(itertools.accumulate(y_set))
        target_y_set = [10,20,50,80,100]
        target_x_set = np.interp(target_y_set, y_set, x_set)
        
        plt.plot(x_set, y_set)
        plt.plot(target_x_set, target_y_set, 'ro')
        plt.vlines(target_x_set, 0,  target_y_set, color='red', linestyle='--', lw=0.2)
        plt.hlines(target_y_set, 0, target_x_set, color='red', linestyle='--', lw=0.2)
        plt.xlabel('area in μ')
        plt.ylabel('cumulative undersize percent')
        plt.show()
    elif numclasses == 3:
        for id in range(1,numclasses):
            pred_id= copy.deepcopy(pred.squeeze(0).numpy().astype(np.uint8))
            pred_id[pred_id!=id] = 0
            pred_id[pred_id==id] = 255
            contours, _= cv2.findContours(pred_id, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            areas = []
            for k in range(len(contours)):
                conmap= np.zeros((pred_id.shape[0],pred_id.shape[1]),dtype=np.uint8)
                cv2.drawContours(conmap, contours, k, 255, -1)
                area = cv2.countNonZero(conmap)
                if area>=5:
                    areas.append(area)

            areas.sort()
            areaSum = sum(areas)
            areaImage = pred_id.shape[0]*pred_id.shape[1]
            avg, std, porosity = tds(areas, areaSum, areaImage)
            print(avg, std, porosity)
            count = Counter(areas)
            percent_dict= {}
            for size, cnt in count.items():
                percent=(size*cnt/areaSum)*100
                percent_dict[size*px] = percent
            x_set = list(percent_dict.keys())
            y_set = list(percent_dict.values())
            y_set = list(itertools.accumulate(y_set))
            target_y_set = [10,20,50,80,100]
            target_x_set = np.interp(target_y_set, y_set, x_set)

            plt.plot(x_set, y_set, colors[id-1])
            plt.plot(target_x_set, target_y_set, 'ro')
            plt.vlines(target_x_set, 0,  target_y_set, color='red', linestyle='--', lw=0.2)
            plt.hlines(target_y_set, 0, target_x_set, color='red', linestyle='--', lw=0.2)
            plt.xlabel('area in μ')
            plt.ylabel('cumulative undersize percent')
            plt.show()

    
    tp, fp, fn, tn = smp.metrics.get_stats(
            pred, gt, mode="multiclass", num_classes=numclasses
        )
    per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
    return per_image_iou, f1_score, accuracy, recall

def visualize(src, pred):
    #  pr_masks[pr_masks==1] = 255
    #  pr_masks[pr_masks==2] = 128
    mask_1 = copy.deepcopy((pred.cpu().numpy()).astype(np.uint8))
    mask_1[mask_1!=1] = 0
    mask_1[mask_1==1] = 255
    contours, _= cv2.findContours(mask_1 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for k in range(len(contours)):
        cv2.drawContours(src, contours, k, (0, 0, 255, 255), 1)

    mask_2 = copy.deepcopy((pred.cpu().numpy()).astype(np.uint8))
    mask_2[mask_2!=2] = 0
    mask_2[mask_2==2] = 255
    contours, _= cv2.findContours(mask_2 , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for k in range(len(contours)):
        conmap= np.zeros((mask_2.shape[0],mask_2.shape[1]),dtype=np.uint8)
        cv2.drawContours(conmap, contours, k, 255, -1)
        area = cv2.countNonZero(conmap)
        min_rect = cv2.minAreaRect(contours[k]) 
        if min(min_rect[1][0], min_rect[1][1])>0:
            rect_ratio = max(min_rect[1][0], min_rect[1][1])/min(min_rect[1][0], min_rect[1][1])
        else:
            print("Warning: min_rect dimensions are zero, skipping ratio calculation.")
        if area <80:
            print("点状Dot")
        elif rect_ratio >2.5:
            print("条状strip")
        else:
            print("块状block")
        
        
        cv2.putText(src, str(area), (int(min_rect[0][0]), int(min_rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0,0), 1)
        cv2.drawContours(src, contours, k, (255, 0, 0, 255), 1)
    return src

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_test_dir', type=str, default='/home/dwu/deepseg/test/Images')
    parser.add_argument('--ckpt_save_path', type=str, default='/home/dwu/deepseg/exp/model/best_ckpt.ckpt')
    parser.add_argument('--test_fig_dir', type=str, default='/home/dwu/deepseg/result_hr')
    args = parser.parse_args()
    
    model = CamVidModel.load_from_checkpoint(
        checkpoint_path= args.ckpt_save_path,
        arch="Unet",
        encoder_name="resnet50",
        in_channels=3,  # Assuming RGB input
        out_classes=3,
    ).cuda()  # Ensure model is moved to GPU


    # Process image sizes
    min_height_train_val, min_width_train_val = process_images_train_val(args.image_test_dir, args.image_test_dir)


    # Create dataset and data loader
    train_dataset = Dataset(
        images_dir= args.image_test_dir,
        masks_dir= args.image_test_dir,
        augmentation=get_training_augmentation(min_height=min_height_train_val, min_width=min_width_train_val),
    )
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True)  # Increase num_workers

    image_paths = train_loader.dataset.images_fps

    def sort_key(path):
        # 从路径中提取文件名，并从文件名中提取数字部分
        filename = path.split('/')[-1]  # 提取文件名
        return int(filename.split('.')[0])  # 返回文件名中的数字部分

    #sorted_images_path = sorted(image_paths, key=sort_key)
    sorted_images_path = image_paths


    for idx, path in enumerate(sorted_images_path):
        image = io.imread(path)
        transform = get_transform(image.shape[:2])
        dataset = SingleImageDataset(path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for images, gt in dataloader:
                output_image_path = os.path.join(args.test_fig_dir, f"{idx}_image.png")
                io.imsave(output_image_path, (255*images.squeeze(0).cpu().numpy().transpose((1,2,0))).astype(np.uint8))
                logits = model(images)
                
                pr_masks = (logits.softmax(dim=1)).argmax(dim=1)
                per_image_iou, f1_score, accuracy, recall = evaluate(pr_masks.cpu(), gt, 3)
                # pr_masks = (logits.softmax(dim=1)).max(1)[1][0]
                # label = visualize(image, pr_masks)
                # output_image_path = os.path.join(args.test_fig_dir, f"{idx}_label_image1.png")
                # io.imsave(output_image_path, label)

                # pr_masks[pr_masks==1] = 255
                # pr_masks[pr_masks==2] = 128
                # output_image_path = os.path.join(args.test_fig_dir, f"{idx}_label_image1.png")
                # io.imsave(output_image_path, (pr_masks.cpu().numpy()).astype(np.uint8))