import torch
import torch.nn.functional as F
import numpy as np
from utils import *
from off_plane_transformation import *
from stnNetwork import stnNet
import Pretrained_FR_Models.irse as irse
import Pretrained_FR_Models.facenet as facenet
import Pretrained_FR_Models.ir152 as ir152
import cv2
from torchvision import transforms
from test_finalcrop import draw_landmark

class Advhat(torch.nn.Module):
    def __init__(self,batch):
        super(Advhat, self).__init__()
        self.device = "cuda:0"
        self.stn = stnNet().to(self.device)
        self.b = batch
        self.alpha1 = np.random.uniform(-1., 1., size=(batch, 1)) / 180. * np.pi
        self.scale1 = np.random.uniform(0.465 - 0.02, 0.465 + 0.02, size=(batch, 1))
        self.y1 = np.random.uniform(-15. - 600. / 112., -15. + 600. / 112., size=(batch, 1))
        self.x1 = np.random.uniform(0 - 600. / 112., 0 + 600. / 112., size=(batch, 1))
        self.alpha2 = np.random.uniform(-1., 1., size=(batch, 1)) / 180. * np.pi
        self.scale2 = np.random.uniform(1. / 1.04, 1.04, size=(batch, 1))
        self.y2 = np.random.uniform(-1., 1., size=(batch, 1)) / 66.
        self.angle = np.random.uniform(17. - 2., 17. + 2., size=(batch, 1))
        self.parab = np.random.uniform(0.0013 - 0.0002,0.0013 + 0.0002, size=(batch, 1))
        self.moments = torch.from_numpy(np.zeros((400, 900, 3)))
        self.loss = []
        self.transform = transforms.Compose([transforms.ToTensor()])



    def re_init(self):
        batch = self.b
        self.loss = []
        self.alpha1 = np.random.uniform(-1., 1., size=(batch, 1)) / 180. * np.pi
        self.scale1 = np.random.uniform(0.465 - 0.02, 0.465 + 0.02, size=(batch, 1))
        self.y1 = np.random.uniform(-15. - 600. / 112., -15. + 600. / 112., size=(batch, 1))
        self.x1 = np.random.uniform(0 - 600. / 112., 0 + 600. / 112., size=(batch, 1))
        self.alpha2 = np.random.uniform(-1., 1., size=(batch, 1)) / 180. * np.pi
        self.scale2 = np.random.uniform(1. / 1.04, 1.04, size=(batch, 1))
        self.y2 = np.random.uniform(-1., 1., size=(batch, 1)) / 66.
        self.angle = np.random.uniform(17. - 2., 17. + 2., size=(batch, 1))
        self.parab = np.random.uniform(0.0013 - 0.0002,0.0013 + 0.0002, size=(batch, 1))
        self.moments = torch.from_numpy(np.zeros((400, 900, 3)))


    def FGSM_with_moments(self,host,target,logo,iter,moments=torch.from_numpy(np.zeros((400, 900, 3)))):
        """
        :param host: tensor [h,w,c] RGB
        :param target: tensor [b,c,h,w] RGB
        :param logo: tensor [b,c,h,w]RGB
        :param iter: 迭代次数
        :param moments: 动量
        :return: tensor 贴纸 和 覆盖贴纸的人脸
        """

        if iter == 100:
            print("Stage:1")
            moments = moments.unsqueeze(0).permute(0,3,1,2).to(self.device)
            moment_val = 0.9
            #step_val = 1. / 51.
            step_val = 0.05
            for epoch in range(iter):
                logo.requires_grad = True
                final_host =draw_landmark(host,logo)
                final_crop = torch.from_numpy(final_host).to(self.device).permute(2,0,1)
                tv_loss = self.tv_loss(logo).to(dtype=torch.float32)
                cos_loss = self.cal_target_loss(final_crop, target,'irse50').to(dtype=torch.float32)
                cos_loss1 = self.cal_target_loss(final_crop, target, "mobile_face").to(dtype=torch.float32)
                cos_loss2 = self.cal_target_loss(final_crop, target, "facenet").to(dtype=torch.float32)
                loss = cos_loss + 0.0001 * tv_loss +cos_loss1+ cos_loss2
                grad_logo = torch.autograd.grad(loss, logo)[0]
                if epoch % 20 == 0:
                    print("COS_LOSS:",cos_loss.data)
                moments = moments * moment_val + grad_logo*(1.-moment_val)
                logo = logo - step_val * moments.sign()
                logo =torch.clamp(logo,min = 0,max = 1 ).detach()
            return logo,moments,host
        #logo [b,c,h,w] moments [b,c,h,w] host [b,c,h,w]
        else:
            print("Stage:2")
            moment_val = 0.995
            #step_val = 1. / 255.
            step_val = 0.01
            for epoch in range(iter):
                logo.requires_grad = True
                final_host = draw_landmark(host, logo)
                final_crop = torch.from_numpy(final_host).to(self.device).permute(2,0,1)

                final_crop.requires_grad = True
                tv_loss = self.tv_loss(logo).to(dtype=torch.float32)
                cos_loss = self.cal_target_loss(final_crop, target,'irse50').to(dtype=torch.float32)
                cos_loss1 = self.cal_target_loss(final_crop, target, "mobile_face").to(dtype=torch.float32)
                cos_loss2 = self.cal_target_loss(final_crop, target, "facenet").to(dtype=torch.float32)
                loss = cos_loss + 0.0001 * tv_loss +cos_loss1+ cos_loss2
                grad_logo = torch.autograd.grad(loss, logo)[0].to(self.device)
                if epoch % 20 == 0:
                    print("COS_LOSS:", cos_loss.data)
                moments = moments * moment_val + grad_logo * (1. - moment_val)
                logo = logo - step_val * moments.sign()
                logo =torch.clamp(logo,min = 0,max = 1 ).detach()
            return logo.squeeze(0),final_crop

    def tv_loss(self,input_t):
        """
        :param input_t: img
        :return: tv损失
        """
        temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
        temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
        temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
        return temp.sum()


    def cos_simi(self, emb_before_pasted, emb_target_img):
        """
        :param emb_before_pasted: 宿主的模型输出结果
        :param emb_target_img: 目标的模型输出结果
        :return: 余弦相似度
        """
        return torch.mean(torch.sum(torch.mul(emb_target_img, emb_before_pasted), dim=1)
                          / emb_target_img.norm(dim=1) / emb_before_pasted.norm(dim=1))


    def cal_target_loss(self, before_pasted, target_img,model_name):
        """
        :param before_pasted: 宿主照片
        :param target_img: 目标照片
        :param model_name: 想要攻击的模型
        :return: 余弦相似度损失
        """
        fr_model = ir152.IR_152((112, 112))
        fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
        input_size = (112, 112)
        if model_name == 'ir152':
            input_size = (112, 112)
            # self.models_info[model_name][0].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/ir152.pth'))
        if model_name == 'irse50':
            self.input_size = (112, 112)
            # self.models_info[model_name][0].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/irse50.pth'))
        if model_name == 'mobile_face':
            input_size = (112, 112)
            # self.models_info[model_name][0].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/mobile_face.pth'))
        if model_name == 'facenet':
            input_size = (160, 160)
            # self.models_info[model_name][0].append((160, 160))
            fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
            fr_model.load_state_dict(torch.load('./Pretrained_FR_Models/facenet.pth'))
        fr_model.to("cuda")
        fr_model.eval()

        before_pasted_resize = F.interpolate(before_pasted.unsqueeze(0), size=input_size, mode='bilinear')
        target_img_resize = F.interpolate(target_img, size=input_size, mode='bilinear')

        # Inference to get face embeddings
        emb_before_pasted = fr_model(before_pasted_resize)
        emb_target_img = fr_model(target_img_resize).detach()

        # Cosine loss computing
        cos_loss = 1 - self.cos_simi(emb_before_pasted, emb_target_img)
        #cos_loss.requires_grad = True
        return cos_loss


    def attach_sticker_onface(self,logo,face,target):
        """
        :param logo: 攻击贴纸 #tensor
        :param face: 宿主人脸 #tensor
        :param target: 目标人脸 #tensor
        :return: host face with logo #tensor
        """
        # Off-Plain Sticker Transformation
        logo_tf = tf.constant(logo.to("cpu").permute(0,2,3,1).detach().numpy(),dtype = tf.float32)
        off_plaine_logo_np = projector(logo_tf,tf.constant(self.parab,dtype = tf.float32).cpu(),tf.constant(self.angle,dtype = tf.float32).cpu()) #numpy [900,900,3]
        off_plaine_logo = self.transform(off_plaine_logo_np).unsqueeze(0).to(self.device) #tensor[1,3,900,900]

        off_plaine_logo = F.interpolate(off_plaine_logo, size=(600,600), mode='bilinear')

        theta = torch.from_numpy(np.array([[np.cos(self.alpha1),np.sin(self.alpha1),-self.x1/450.],[-np.sin(self.alpha1),np.cos(self.alpha1),-self.y1/450.]])).float().to(self.device)
        prepared1 = self.stn(off_plaine_logo,theta)


        prepared = off_plaine_logo.permute(0, 2, 3, 1)
        mask_input = prepared1.permute(0, 2, 3, 1)
        face =face.permute(0, 2, 3, 1)

        # Union of the sticker and face image
        theta2 = torch.from_numpy(np.array([[np.cos(self.alpha2),np.sin(self.alpha2),-self.x1/450.],[-np.sin(self.alpha2),np.cos(self.alpha2),-self.y1/450.]])).to(self.device).float()
        #united = prepared * mask_input + face * (1 - mask_input)
        united = prepared[:, 300:, 150:750]  + face * (1 - mask_input[:, 300:, 150:750])
        img_tmp = united.permute(0,3,1,2)
        img_tmp1 = self.stn(img_tmp,theta2)
        final_crop = torch.clamp(img_tmp1, 0., 1.)

        return final_crop








