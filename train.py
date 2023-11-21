import torch
import tensorflow as tf
import numpy as np
from advhat import Advhat
from PIL import Image
from torchvision import transforms
import skimage.io as io
from utils import *
import os
device = "cuda:0"
if __name__ == "__main__":
    host_dir = "./before_aligned_600"
    #host_dir = "./dataset"
    target_dir = "./target_aligned_600/Camilla_Parker_Bowles_0002.jpg"
    target_img = io.imread(target_dir)/255.0


    logo_dir = "./logo/example.png"
    logo_img = io.imread(logo_dir)/255.0
    #logo_img = np.random.rand(400, 900, 3).astype(np.float32)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize([600,600])])
    transform1 = transforms.Compose([transforms.ToTensor()])

    file_list = os.listdir(host_dir)
    model = Advhat(1).to(dtype=torch.float32,device=device)
    target_img = torch.from_numpy(target_img).unsqueeze(0).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    for file in file_list:
        path = os.path.join(host_dir,file)
        host_img = io.imread(path)
        host_img = torch.from_numpy(host_img).to(device=device,dtype=torch.float32)

        logo_tensor = torch.from_numpy(logo_img).to(device=device,dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)


        logo,moments,host = model.FGSM_with_moments(host_img,target_img,logo_tensor,100)
        moments = moments.to(device)
        logo,final_result = model.FGSM_with_moments(host_img,target_img,logo,200,moments)
        name = file.split(".")

        logo_numpy = logo.cpu().permute(1,2,0).detach().numpy().astype(np.float32)*255
        final_numpy = final_result.permute(1,2,0).cpu().detach().numpy().astype(np.float32)*255
        logo_numpy = cv2.cvtColor(logo_numpy,cv2.COLOR_RGB2BGR).astype(np.uint8)
        final_numpy = cv2.cvtColor(final_numpy, cv2.COLOR_RGB2BGR).astype(np.uint8)

        cv2.imwrite("./face_with_logo_ir152/" + name[0] + ".png",  final_numpy)
