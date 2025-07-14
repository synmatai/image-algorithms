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
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import argparse
from PIL import Image
import glob
from skimage import io, color, measure
import tifffile as tif
from albumentations import Compose, LongestMaxSize, PadIfNeeded
import torch.nn.functional as F
from torchvision import transforms as torchtrans


class SingleImageDataset(BaseDataset):
    def __init__(self, image_path, transform):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # 只有一张图片

    def __getitem__(self, idx): 
        image = io.imread(self.image_path)
        if image.shape[2] == 4:  # 检查是否是 RGBA 图像
            image = image[:, :, :3]  # 去掉 Alpha 通道
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        image = torchtrans.ToTensor()(image)
        return image

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
        # mask[mask!=255] = 0
        mask[mask==255] = 1
        mask[mask==128] = 2
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        image = torchtrans.ToTensor()(image)
        # return image, mask_remap
        return image, mask

    def __len__(self):
        return len(self.ids)

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


def get_validation_augmentation(min_height=448,min_width=608):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height, min_width),
    ]
    return A.Compose(test_transform)

def process_images_train_val(train_image_path,val_image_path):
    max_width, max_height = 0,0
    files = glob.glob(train_image_path+'/*')
    print(files)
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

def process_images_test(test_image_path):
    """
    读取文件夹下的所有图片，检查图片的长宽是否能被32整除。
    如果不能，则扩充到最近的32的倍数，并返回所有扩充后图片的最大长宽。

    :param folder_path: 文件夹路径
    :return: (max_width, max_height) 扩充后的最大长宽
    """
    max_width, max_height = 0,0
    files = glob.glob(test_image_path+'/*')
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
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255) #smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.cnt = 0

    def forward(self, image):
        # Normalize image
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
        loss = self.loss_fn(logits_mask, mask.long())

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        metrics = {
            f"{stage}_loss": loss,
        }

        self.log_dict(metrics, prog_bar=True)


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

def rgb_string_to_tuple(rgb_string):
    """将 RGB 字符串转换为元组."""
    # 提取数字并转换为整数。
    rgb_values = list(map(int, rgb_string[4:-1].split(", ")))
    return tuple(rgb_values)

def visualize(label_image, save_path, colors, alpha=0.3):
    """
    使用自定义颜色映射将标签图像转换为RGB图像。

    Args:
        label_image: 标签图像。
        colors: 颜色字典，key为label，value为颜色。
        alpha: 混合原始图像的透明度。
    """
    # 将标签图像转换为整数类型
    label_image = label_image.astype(int)
    
    # 创建输出图像的数组
    out = np.zeros(label_image.shape + (3,), dtype=np.uint8)

    for label in np.unique(label_image):
        print("Processing label:", label)
        mask = label_image == label 
        
        if label in colors:
            # 将颜色字符串转换为 RGB 元组
            out[mask] = rgb_string_to_tuple(colors[label])  # 使用 label 直接索引

    # 调试输出

    io.imsave(save_path + "preview.png", out.astype(np.uint8))  # 保存图像

if __name__=='__main__':
    #指定数据集位置,image为原图片，mask是保存的label_maps
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_train_dir',type=str,default='/home/dwu/FASNet-main/CGFSDS-9/D0047/Images') #切割后的图片
    parser.add_argument('--mask_train_dir',type=str,default='/home/dwu/FASNet-main/CGFSDS-9/D0047/GT') 
    parser.add_argument('--image_val_dir',type=str,default='/home/dwu/FASNet-main/CGFSDS-9/D0047/Images')
    parser.add_argument('--mask_val_dir',type=str,default='/home/dwu/FASNet-main/CGFSDS-9/D0047/GT')
    #模型保存路径，测试图片保存路径，类别，轮数
    parser.add_argument('--ckpt_save_dir',type=str,default='/home/dwu/deepseg/exp/model')
    parser.add_argument('--test_fig_dir',type=str,default='/home/dwu/deepseg/result') #输出目录
    parser.add_argument('--outclass',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=500)#400
    
    args = parser.parse_args()

    tif_files = os.listdir(args.mask_train_dir) 

    for file in tif_files:
        if file.endswith('.tif'):
            image_path = os.path.join(args.mask_train_dir, file)
            image = tif.imread(image_path)
            image = image.astype(np.uint8)  # 确保数据类型为8位无符号整数
            io.imsave(image_path.replace("tif","png"), image)  # 保存为8位无符号整数的TIFF文件
            # 确保图像是8位无符号整数

    #获取不同数据集照片的最大扩充长宽
    min_height_train_val,min_width_train_val = process_images_train_val(args.image_train_dir,args.image_val_dir)
    #创建数据集
    train_dataset = Dataset(
        args.image_train_dir,
        args.mask_train_dir,
        augmentation=get_training_augmentation(min_height=min_height_train_val,min_width=min_width_train_val),
    )

    valid_dataset = Dataset(
        args.image_val_dir,
        args.mask_val_dir,
        augmentation=get_validation_augmentation(min_height=min_height_train_val,min_width=min_width_train_val),
    )


    # Change to > 0 if not on Windows machine
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    EPOCHS = args.epochs
    T_MAX = EPOCHS * len(train_loader)
    # Always include the background as a class
    # OUT_CLASSES = len(train_dataset.CLASSES)
    
    #模型构建
    model = CamVidModel("Unet", "resnet50", in_channels=3, out_classes=args.outclass)    
    
    #早停
    early_stop_callback = EarlyStopping(monitor="valid_dataset_iou", mode="max", patience=20)
    
    #保存最优模型
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_dataset_iou',  # 监控的指标
        dirpath=args.ckpt_save_dir,  # 保存路径
        filename='best_ckpt',  # 文件名格式
        save_top_k=1,  # 保存的最佳模型数量
        mode='max'  # 监控指标的最小化
    )
    
    #训练
    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1, callbacks=[checkpoint_callback]) #, early_stop_callback
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    #加载最优模型
    best_model_path = checkpoint_callback.best_model_path
    model = CamVidModel.load_from_checkpoint(
    checkpoint_path=best_model_path,
    arch="Unet",
    encoder_name="resnet50",
    in_channels=3,
    out_classes=args.outclass,
    )

    #验证集验证
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print(valid_metrics)
    
      
    
    #固定颜色
    colors = {
    0: "rgb(35, 155, 86)",    # 绿色
    1: "rgb(231, 76, 60)",     # 红色
    2: "rgb(93, 173, 226)",    # 蓝色
    3: "rgb(255, 204, 0)",     # 黄色
    4: "rgb(145, 61, 244)",    # 紫色
    5: "rgb(255, 127, 80)",     # 桃色
    6: "rgb(0, 153, 255)",      # 天蓝色
    7: "rgb(255, 51, 153)",     # 粉红色
    8: "rgb(0, 204, 0)",        # 深绿色
    9: "rgb(238, 117, 0)"       # 橙色
    }
    # Switch the model to evaluation mode

    #image_paths = glob.glob(os.path.join(args.image_train_dir, '*.[pjg][np][ge]*'))  # 匹配所有常见图像扩展名
     
    image_paths = train_loader.dataset.images_fps

    def sort_key(path):
        # 从路径中提取文件名，并从文件名中提取数字部分
        filename = path.split('/')[-1]  # 提取文件名
        return int(filename.split('.')[0])  # 返回文件名中的数字部分
    
    sorted_images_path = image_paths#sorted(image_paths, key=sort_key)
    

    for idx, path in enumerate(sorted_images_path):
        image = io.imread(path)
        transform = get_transform(image.shape[:2])
        dataset = SingleImageDataset(path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for images in dataloader:
                output_image_path = os.path.join(args.test_fig_dir, f"{idx}_image.png")
                io.imsave(output_image_path, (255*images.squeeze(0).cpu().numpy().transpose((1,2,0))).astype(np.uint8))
                
                # #images = images.permute(0, 3, 1, 2)  # 调整维度 (batch, channels, height, width)
                # #print("Transposed shape:", images.shape)

                # pad_h = (32 - images.shape[2] % 32) % 32
                # pad_w = (32 - images.shape[3] % 32) % 32

                # images = F.pad(images, (0, pad_w, 0, pad_h))  # 进行必要的填充

                logits = model(images)  # 通过模型获取结果olkp c
                pr_masks = (logits.softmax(dim=1)).max(1)[1][0]
                pr_masks[pr_masks==1] = 255
                pr_masks[pr_masks==2] = 128
                output_image_path = os.path.join(args.test_fig_dir, f"{idx}_label_image.png")
                io.imsave(output_image_path, (pr_masks.cpu().numpy()).astype(np.uint8))

                # visualize(pr_masks[0, 0,:,:].cpu().numpy(), 
                #           os.path.join(args.test_fig_dir, f"{idx}_"), 
                #           colors, 
                #           alpha=0.8) 





 
#    # 测试结果可视化
#    # for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
#    for idx, (image, pr_mask) in enumerate(zip(images, pr_masks)):
#        if idx <= 4:  # Visualize first 5 samples
#            plt.figure(figsize=(12, 6))
#
#            # Original Image
#            plt.subplot(1, 2, 1)
#            plt.imshow(
#                image.cpu().numpy().transpose(1, 2, 0)
#            )  # Convert CHW to HWC for plotting
#            plt.title("Image")
#            plt.axis("off")
#
#
#            # Predicted Mask
#            plt.subplot(1, 2, 2)
#            plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")  # Visualize predicted mask
#            plt.title("Prediction")
#            plt.axis("off")
#
#            # Show the figure
#            plt.show()
#            save_path = os.path.join(args.test_fig_dir, f"test_{idx}.png")
#            plt.savefig(save_path)
#        else:
#            break