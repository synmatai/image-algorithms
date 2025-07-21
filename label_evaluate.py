import cv2
import tifffile as tf
import json
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm, halfnorm
import pandas as pd

from collections import Counter
import itertools


def eps_enclosing_circle(contour):
    (_, _), radius = cv2.minEnclosingCircle(np.array(contour))
    return 2*np.pi*radius
def least_edge(contour, epsilon):
    approx = cv2.approxPolyDP(np.array(contour), epsilon, True)
    return approx.shape[0]

def shape_class(annotations):
    bound1 = annotations["objects"][36]["segmentation"]
    bound2 = annotations["objects"][25]["segmentation"]
    bound1 = [[int(a[0]), int(a[1])] for a in bound1]
    bound2 = [[int(a[0]), int(a[1])] for a in bound2]

    mask = np.zeros(shape=(annotations["info"]["height"], annotations["info"]["width"]), dtype=np.uint8)
    # mask = cv2.drawContours(mask, [approx], -1, 255, 1) 
    # cv2.fillPoly(mask, [np.array(bound1).reshape(-1, 1, 2)], 255)
    # cv2.fillPoly(mask, [np.array(bound2).reshape(-1, 1, 2)], 255)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # meth1:计算边数目
    userset_edge_num = 10   #用户设定边数目大于等于多少算是圆形
    epsilon = 0.01*cv2.arcLength(np.array(bound1), True)
    least_edge_num = least_edge(bound1, epsilon)
    if least_edge_num >= userset_edge_num:
        print("bound1 is circle-like, edge num:", least_edge_num)
    else:
        print("bound1 is not circle-like, edge num:", least_edge_num)
  

    # meth2:计算每个向量的角度（0~360度）
    angles1 = []  
    for i in range(0, len(bound1)):
        vec = [bound1[i][0] - bound1[(i+1)%len(bound1)][0], bound1[i][1] - bound1[(i+1)%len(bound1)][1]]
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if angle < 0:
            angle += 360
        angles1.append(angle)
    
    angles2 = []  
    for i in range(0, len(bound2)):
        vec = [bound2[i][0] - bound2[(i+1)%len(bound2)][0], bound2[i][1] - bound2[(i+1)%len(bound2)][1]]
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if angle < 0:
            angle += 360
        angles2.append(angle)
    
    # 绘制直方图
    # fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    # axs[0].hist(angles1, bins=12, range=(0, 360), edgecolor='blue')
    # axs[0].set_xlabel('Angle (degrees)')
    # axs[0].set_ylabel('Count')
    # axs[1].hist(angles2, bins=12, range=(0, 360), edgecolor='blue')
    # axs[1].set_xlabel('Angle (degrees)')
    # axs[1].set_ylabel('Count')
    # plt.show()

def cal_contact_ratio(bound1, bound2, height, width):
    mask = np.zeros(shape=(height, width), dtype=np.uint8)
    mask = cv2.drawContours(mask, [np.array(bound2).astype(np.int32)], 0, 255, -1) 
    
    touching_num = 0
    for pt in bound1:
        nebs = [[pt[0]+off[0], pt[1]+off[1]] for off in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                 if 0 <= pt[0]+off[0] < width and 0 <= pt[1]+off[1] < height]
       
        for neb in nebs:
            if mask[int(neb[1]), int(neb[0])] == 255:
                touching_num += 1
                break
    contact_ratio = touching_num / len(bound1)
    return contact_ratio

def contact_distribution(annotations):
    num_contours = len(annotations["objects"])
    contact_matrix = np.zeros((num_contours, num_contours), dtype=np.float32)
    for i in range(num_contours):
        pti = annotations["objects"][i]["bbox"]
        area = annotations["objects"][i]["area"]
        dist_threshold = 3*2 * np.sqrt(area / np.pi)
        for j in range(num_contours):
            if i == j:
                continue
            ptj = annotations["objects"][j]["bbox"]
            dist = np.linalg.norm(np.array([pti[0], pti[1]]) - np.array([ptj[0], ptj[1]]))
            if dist < dist_threshold:
                contact_ratio = cal_contact_ratio(annotations["objects"][i]["segmentation"],
                                                      annotations["objects"][j]["segmentation"], 
                                                      annotations["info"]["height"],  
                                                      annotations["info"]["width"])
                contact_matrix[i][j] = contact_ratio
            else:
                continue
    contact_list = ((contact_matrix.reshape(-1)*100).astype(np.int32)).tolist()
    no_zero_contact_list = [i for i in contact_list if i != 0]

    fig, ax = plt.subplots(1, 1)
    n, bins, patches = ax.hist(no_zero_contact_list, bins=max(no_zero_contact_list)-min(no_zero_contact_list)+1, density=True, 
                                                                alpha=0.5, edgecolor='black', label='hist')
   
    ax.set_xlim(min(no_zero_contact_list), max(no_zero_contact_list)*3)
    ax.set_xticks(np.append(np.arange(min(no_zero_contact_list), max(no_zero_contact_list)+1, 10), np.arange(max(no_zero_contact_list)+1, max(no_zero_contact_list)*3, 20)))
    ax.set_yticks(np.arange(0.0, 0.6, 0.01))
    ax.set_xlabel('Contact Ratio %')
    ax.set_ylabel('Density')
    
    paras = halfnorm.fit(no_zero_contact_list)
    ax.plot(bins, halfnorm.pdf(bins, paras[0],paras[1]), linestyle='--', color='b', linewidth=2, label='halfnorm pdf')
    plt.title('Contact Ratio Distribution')
    plt.legend()
    plt.show()

def dia_distribution(annotations):
    num_objs = len(annotations["objects"])
    dia_list = []
    for i in range(num_objs):
        bw, bh = annotations["objects"][i]["bbox"][2], annotations["objects"][i]["bbox"][3]
        dia = (bw + bh) / 2
        dia_list.append(dia)    fig, ax = plt.subplots(1, 1)
    n, bins, patches = ax.hist(dia_list, bins=30, density=True, range = (0, 1200),
                                                                alpha=0.5, edgecolor='black', label='hist')

    
    mean, std = np.mean(dia_list), np.std(dia_list)    
    ax.plot(bins, norm.pdf(bins, mean, std), linestyle='--', color='b', linewidth=2, label='pdf')
    ax.set_xlabel('Diameter (pix)')
    ax.set_ylabel('Density')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(bins, norm.cdf(bins, mean, std), linestyle='--', color='g', linewidth=1, label='cdf')
    ax2.set_ylabel('Cumulative Density')
    ax2.legend()
    plt.title('Size Diameter Distribution')
    plt.show()
    
    
    

def main():
    with open('D:/data/example/Label/2023082801-100L-4.json', 'r') as file:
        annotations = json.load(file)
        #contact_distribution(annotations)
        dia_distribution(annotations)
        

        #shape_class(annotations)

    # datas = pd.read_csv("C:/Users/72983/Desktop/Results.csv", encoding='GBK')
    # print("数据读取成功：")
    # areas = []
    # for i in datas['Area']:
    #     areas.append(i)
    # #px = 0.00476562
    # px = 0.0046875
    # areas.sort()
    # areaSum = sum(areas)
    # count = Counter(areas)
    # percent_dict= {}
    # for size, cnt in count.items():
    #     percent=(size*cnt/areaSum)*100
    #     percent_dict[size*px] = percent
    # x_set = list(percent_dict.keys())
    # y_set = list(percent_dict.values())
    # y_set = list(itertools.accumulate(y_set))
    # target_y_set = [10,20,50,80,100]
    # target_x_set = np.interp(target_y_set, y_set, x_set)
    # plt.plot(x_set, y_set, 'blue')
    # plt.plot(target_x_set, target_y_set, 'ro')
    # plt.vlines(target_x_set, 0,  target_y_set, color='red', linestyle='--', lw=0.2)
    # plt.hlines(target_y_set, 0, target_x_set, color='red', linestyle='--', lw=0.2)
    # plt.xlabel('area in μ')
    # plt.ylabel('cumulative undersize percent')
    # plt.show()



    # cur_path = "C:/Users/72983/Downloads/D0047_mask.tiff" 
    # #img_tf = cv2.imread(cur_path)
    # img_tf = tf.imread(cur_path)
    # img_tf[img_tf==1]=255
    # img_tf[img_tf==2]= 128
    # img_tf = img_tf.astype(np.uint8)
    # # crop_list = [[0, 288, 0, 384], [0,288, 320, 704], [192, 480, 0, 384], [192, 480, 320, 704], [144, 432, 192, 576]]
    # # for i in range(len(crop_list)):
    # #     img_crop = img_tf[crop_list[i][0]:crop_list[i][1], crop_list[i][2]:crop_list[i][3]]
    # #     cv2.imwrite(cur_path.replace('D0047_mask.tiff', 'D0047_crop'+str(i)+".png"), img_crop)
    
    
    # cv2.imwrite("C:/Users/72983/Downloads/D0047_mask.png", img_tf)


    # folder = "C:/Users/72983/Desktop/result_hr"
    # for file in os.listdir(folder):
    #     if os.path.splitext(file)[0] == "2_image":
    #         image = cv2.imread(os.path.join(folder, file))
    #         mask = cv2.imread(os.path.join(folder, "2_label_image1.png"), 0)
    #         contours, _= cv2.findContours(mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         for k in range(len(contours)):
    #             result = cv2.drawContours(image, contours, k, (255,0,0),1)


    #         save_name ="./" + file
    #         cv2.imwrite(save_name,result)

    

if __name__ == '__main__':
    main()