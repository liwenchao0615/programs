'''
Author       : WenChan Li
Date         : 2024-03-18 22:22:55
LastEditors  : WenChan Li
LastEditTime : 2024-03-19 16:54:18
Description  : 
Copyright 2024 OBKoro1, All Rights Reserved. 
2024-03-18 22:22:55
'''
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import argparse
import os
from torchvision import transforms
from model.ZSIEC import LSIENet
from loss.Myloss import L_spa, L_exp, L_ism,L_spa8,L_TV,ContrastLoss,Loss_exp, IlluminationSmoothLoss
import numpy as np
from PIL import Image
import shutil

import torch.nn.functional as F

def weights_init(m):
    # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    if isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=0.1)
        # nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


# def alpha_delta_torch(x):
#     return x * 5 * torch.exp((x ** 1.6) * -14)

# def alpha_torch(x, phs, phh):
#     return x + phs * alpha_delta_torch(x) - phh * 0.2 * alpha_delta_torch(1 - x)



# def data_term(res, ori):
#     return torch.mul(torch.sign(res - 0.5), res - ori)
#
#
# def block_layer(img, blockSize):
#     return F.avg_pool2d(img, kernel_size=blockSize, stride=blockSize, padding=0)
#
#
#
# def block_contrast(img, oriSize, padRow, padCol, kernelSize, kernel):
#     imgpad = F.pad(img.view(1, 1, oriSize, oriSize), (padCol, padCol, padRow, padRow), mode='reflect')
#
#     value = torch.FloatTensor(kernel).view(1, 1, kernelSize[0], kernelSize[1]).cuda()
#     conv_layer = nn.Conv2d(1, 1, kernel_size=kernelSize, stride=1, padding=0, bias=False).cuda()
#     conv_layer.weight = nn.Parameter(value)
#     output = conv_layer(imgpad)
#
#     return output
#
#
# def block_smooth_term(oriImgBlockContrast, newImgBlockContrast):
#     return (oriImgBlockContrast - newImgBlockContrast).pow(2)
#
# def compute_energy_function(oriImg,newImg, paraSize):
#
#     # Block layer
#     oriImgBlock = block_layer(oriImg, blockSize=4)
#     newImgBlock = block_layer(newImg, blockSize=4)
#
#     # dataTerm
#     Edata = torch.sum(data_term(newImgBlock, oriImgBlock))
#
#     # Smooth term
#     oriImgBlockContrast_right = block_contrast(oriImgBlock, oriSize=paraSize, padRow=0, padCol=1, kernelSize=[1, 3], kernel=[0, 1, -1])
#     newImgBlockContrast_right = block_contrast(newImgBlock, oriSize=paraSize, padRow=0, padCol=1, kernelSize=[1, 3], kernel=[0, 1, -1])
#     smoothTerm_right = block_smooth_term(oriImgBlockContrast_right, newImgBlockContrast_right)
#
#     oriImgBlockContrast_left = block_contrast(oriImgBlock, oriSize=paraSize, padRow=0, padCol=1, kernelSize=[1, 3], kernel=[-1, 1, 0])
#     newImgBlockContrast_left = block_contrast(newImgBlock, oriSize=paraSize, padRow=0, padCol=1, kernelSize=[1, 3], kernel=[-1, 1, 0])
#     smoothTerm_left = block_smooth_term(oriImgBlockContrast_left, newImgBlockContrast_left)
#
#     oriImgBlockContrast_up = block_contrast(oriImgBlock, oriSize=paraSize, padRow=1, padCol=0, kernelSize=[3, 1], kernel=[-1, 1, 0])
#     newImgBlockContrast_up = block_contrast(newImgBlock, oriSize=paraSize, padRow=1, padCol=0, kernelSize=[3, 1], kernel=[-1, 1, 0])
#     smoothTerm_up = block_smooth_term(oriImgBlockContrast_up, newImgBlockContrast_up)
#
#     oriImgBlockContrast_down = block_contrast(oriImgBlock, oriSize=paraSize, padRow=1, padCol=0, kernelSize=[3, 1], kernel=[0, 1, -1])
#     newImgBlockContrast_down = block_contrast(newImgBlock, oriSize=paraSize, padRow=1, padCol=0, kernelSize=[3, 1], kernel=[0, 1, -1])
#     smoothTerm_down = block_smooth_term(oriImgBlockContrast_down, newImgBlockContrast_down)
#
#     Esmooth = 0.5 * (smoothTerm_right + smoothTerm_left + smoothTerm_up + smoothTerm_down).sum()
#
#     # Energy function
#     En = Edata + 12 * Esmooth
#
#     return En


def alpha_delta_numpy(x):
    return x * 5 * np.exp((x ** 1.6) * -14)


def alpha_numpy(x, phs, phh):
    return x + phs * alpha_delta_numpy(x) - phh * alpha_delta_numpy(1 - x)

def enhance(config):


    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Input Preprocessing
    net = LSIENet()
    # net = net.to(config.device)
    net = net.eval().to(config.device)


    input_folders = os.listdir(config.data_input)

    for input_folder in input_folders:
        input_folder_path = os.path.join(config.data_input, input_folder)
        output_folder_path = os.path.join(config.result, input_folder)
        os.makedirs(output_folder_path, exist_ok=True)
        # 添加子文件夹路径
        output_folder_path_enhanced = os.path.join(output_folder_path, "enhanced_results")
        os.makedirs(output_folder_path_enhanced, exist_ok=True)

        output_folder_path_Y = os.path.join(output_folder_path, "Y_results")
        os.makedirs(output_folder_path_Y, exist_ok=True)

        enhanced_images = []


        image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('jpg', 'jpeg', 'png','JPG'))]

        for image_file in image_files:
            image_path = os.path.join(input_folder_path, image_file)
            img = cv2.imread(image_path)
            # img = cv2.resize(img, (512, 512))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            B = img[:, :, 0]
            G = img[:, :, 1]
            R = img[:, :, 2]
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            I = 0.596 * R - 0.274 * G - 0.322 * B
            Q = 0.211 * R - 0.523 * G + 0.312 * B
            base_layer = cv2.ximgproc.guidedFilter(Y.astype('uint8'), Y.astype('uint8'),
                                                   int(0.04 * min(512, 512)), 800)
            Y0 =cv2.resize(Y/255, (512, 512))
            # Y0 = Y / 255
            Y0_imgrev = 1 - Y0

            imgV= transforms.ToTensor()(Y0).float().unsqueeze(0).to(config.device)
            imgV_imgrev = transforms.ToTensor()(Y0_imgrev).float().unsqueeze(0).to(config.device)

            images_to_train = [imgV, imgV_imgrev]
            exp_loss= Loss_exp(4)
            Noise_Loss= ContrastLoss(128)
            # loss_spa = L_spa8(4)
            # Ill=IlluminationSmoothLoss(config.device)

            L_TVv = L_TV()
            lossspa = L_spa8(4)
            # lossillu = L_ism()
            # lossexp = L_exp(16)

            optimizer = optim.Adam(net.parameters(), lr=config.lr ,weight_decay=config.weight_decay)

            # Model Iteration
            for img_idx in images_to_train:
                for i in range(config.num_epochs+1):
                    paraS,paraH, newImg  = net(img_idx)
                    # newImg = alpha_torch(imgV, paraS, paraH)
                    # En = compute_energy_function(img_idx, newImg, paraSize=128)

                    # loss computing
                    lossexp=exp_loss(img_idx,newImg)
                    lossnoise= 16*Noise_Loss(img_idx,newImg)
                    # lossspa=torch.mean(loss_spa(imgV,newImg))

                    # loss_exp =  10*torch.mean(lossexp(newImg, 0.5))
                    # loss_illu=Ill(newImg)
                    loss_spa = 1*torch.mean(lossspa(img_idx,newImg))
                    # loss_illu = lossillu(imgV, newImg)
                    loss_LTV =1600*L_TVv(newImg)


                    # loss computing
                    loss = loss_spa  + lossexp+ loss_LTV + lossnoise
                    # loss = En+loss_LTV+loss_spa

                    # backward
                    net.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # log
                    if i % 100 == 0:
                        print("Loss at iteration", i, ":", loss.item(), '  loss_exp ',    lossexp.item(), 'loss_spa', loss_spa.item(),'loss_LTV', loss_LTV.item())

                new_base = alpha_numpy(base_layer / 255, np.float64(paraS.item()), np.float64(paraH.item()))
                new_base_path = os.path.join(output_folder_path_Y, f"new_base_{image_file}")
                new_base_image = Image.fromarray(np.uint8(new_base * 255))
                new_base_image.save(new_base_path)
                # Calculate new brightness value
                ratio = new_base / (base_layer / 255 + 0.00001)

                newY = ratio * Y
                newI = ratio * I
                newQ = ratio * Q

                # 计算新的R、G、B值
                newR = (1.0 * newY + 0.956 * newI + 0.621 * newQ).clip(0, 255)
                newG = (1.0 * newY - 0.272 * newI - 0.647 * newQ).clip(0, 255)
                newB = (1.0 * newY - 1.106 * newI + 1.703 * newQ).clip(0, 255)

                # 将新的颜色值应用到图像的RGB通道上
                result = img.copy()
                result[:, :, 0] = newB
                result[:, :, 1] = newG
                result[:, :, 2] = newR

                enhanced_images.append(result)

            forwardimg, reverseimg = enhanced_images[-2:]
            # fusion input image, enhanced image and suppressed image to generate final results
            mergecore = cv2.createMergeMertens(1, 1, 1)
            img = mergecore.process([img, forwardimg,reverseimg])
            new_baseS = img*255





            # Save the enhanced result
            # enhanced_image = Image.fromarray(result)
            # enhanced_image.save('./results/output/test/enhanced_re.jpg')
            # scaled_newY = np.uint8(new_base * 255)
            # enhanced_newY_image = Image.fromarray(scaled_newY)
            # # enhanced_newY = Image.fromarray(newY)
            enhanced = np.uint8(np.clip(new_baseS, 0, 255))
            enhanced_image = Image.fromarray(enhanced)
            enhanced_image.save(os.path.join(output_folder_path, f"enhanced_{image_file}"))
            Y0_path = os.path.join(output_folder_path_Y, f"Y0_{image_file}")
            Y0_imgrev_path = os.path.join(output_folder_path_Y, f"Y0_imgrev_{image_file}")
            enhanced_Y_path = os.path.join(output_folder_path_Y, f"enhanced_Y_{image_file}")
            enhanced_Y_imgrev_path = os.path.join(output_folder_path_Y, f"enhanced_Y_imgrev_{image_file}")

            # 保存 Y0, Y0_imgrev 和他们的增强结果图像
            Y0_image = Image.fromarray(np.uint8(Y0 * 255))
            Y0_image.save(Y0_path)

            Y0_imgrev_image = Image.fromarray(np.uint8(Y0_imgrev * 255))
            Y0_imgrev_image.save(Y0_imgrev_path)

            enhanced_Y_image = Image.fromarray(forwardimg)
            enhanced_Y_image.save(enhanced_Y_path)

            enhanced_Y_imgrev_image = Image.fromarray(reverseimg)
            enhanced_Y_imgrev_image.save(enhanced_Y_imgrev_path)

            print(f"Enhanced result saved at: {output_folder_path_enhanced}")
            print(f"Y0 and Y0_imgrev saved at: {output_folder_path_Y}")
            enhanced_images.clear()




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_input', type=str, default="data/input_data/test/")
    parser.add_argument('--result', type=str, default="results/output/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default='1000')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--imgSize', type=int, default='128')


    config = parser.parse_args()


    enhance(config)


