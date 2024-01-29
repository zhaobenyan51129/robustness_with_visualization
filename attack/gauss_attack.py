import matplotlib.pyplot as plt
import torch
import numpy as np
from models.vit_model import vit_base_patch16_224
import os

vit_b_16 = vit_base_patch16_224()
# 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
weights_path = "/home/zhaobenyan/Attack_robustness/vit_model/files/vit_base_patch16_224.pth"
vit_b_16.load_state_dict(torch.load(weights_path, map_location="cpu"))

def get_pretrained_vit_b_16_intermediates(vit_model, X, encoder_num): # vit_model: vit_b_16, X: [b, 3, 224, 224]
      """
      获取vit模型的embedding层, 所有encoder层, 输出层; 注意该操作对显存高占用
      return: 
            embedding_output: [batch, num_patches, d_model] 
            vit_intermediates: [num_encoders, batch, num_patches+1(class_token), d_model]
            output:[batch, 1000]
      """
      vit_intermediates = []  # 构建储存所有中间层值的列表
      # 构造hook函数
      def forward_hook(module, input, output):
            vit_intermediates.append(output)

      vit_b_16.patch_embed.register_forward_hook(forward_hook) # 获取conv_proj层的输出[b, 768, 14, 14], 其中768是d_model
      # 监控所有Encoder层
      for encoder_idx in range(encoder_num):
            getattr(vit_model.blocks, f"{encoder_idx}").register_forward_hook(forward_hook)
      output = vit_model(X)
      embedding_output = vit_intermediates.pop(0).reshape([1, vit_b_16.embed_dim, -1]).permute(0, 2, 1) + vit_b_16.pos_embed[:, 1:, :]
      intermediate_output=torch.cat([vit_intermediate.unsqueeze(0) for vit_intermediate in vit_intermediates], dim=0)
      
      return embedding_output, intermediate_output, output
    
def vit_pack(data, encoder): # numpy [1, 224, 224, 3]
    data = torch.tensor(data).permute(0, 3, 1, 2) # [1, 3, 224, 224]
    data = data.to(torch.float32)
    embedding_output, vit_intermediates, output = get_pretrained_vit_b_16_intermediates(vit_b_16, data, encoder_num=encoder)
    vit_intermediates_list=[]
    for i in range(encoder):
        vit_intermediates_encoder=vit_intermediates[i][:, 1:, :].reshape(-1).detach().numpy()
        vit_intermediates_list.append(vit_intermediates_encoder)
    intermediates_output = np.stack(vit_intermediates_list)
    return embedding_output.reshape(-1).detach().numpy(),intermediates_output, output.reshape(-1).detach().numpy()

class GaussianNoiseTestVit:

      def __init__(self, input_fig, encoder_num, model_str='vit_b_16',): # input: [h, w, 3], numpy
            self.encoder_num=encoder_num
            if input_fig.ndim ==3:
                  self.input_fig = input_fig[np.newaxis, :] #[h,w,3]->[1,224,224,3]
            else:
                  self.input_fig = input_fig
            self.img_size = self.input_fig.shape[-2]
            self.model = vit_pack
            self.plot_color = 'b'

            self.unattacked_output = self.model(self.input_fig,encoder_num) # [150528,], [encoder_num,150528],[1000,]
            print('初始output大小', self.unattacked_output[0].shape, self.unattacked_output[1].shape,self.unattacked_output[2].shape)
    
      def get_noised(self, delta_list,  grating=True, fix=True): 
            '''
            grating设置为TRUE的时候只攻击前两个channel
            fix:固定白噪声的pattern
                  fix=True:先生成(0,1)正态分布standard_preturb,然后对每个delta,
                  用delta乘这个standard_preturb作为噪声（每个噪声值相差一个倍数）
                  fix=False:每个delta都随机生成一个（0，delta）的噪声
            '''
            self.delta_list = delta_list
            noise=torch.load('/home/zhaobenyan/Attack_robustness/vit_model/files/normal_noise.pth') #噪声，需手动修改

            # 生成被白噪声攻击过的图片list
            if fix:
                  standard_preturb = noise[np.newaxis, :]  #[224,224,3]->[1,224,224,3]
                  if grating:
                        standard_preturb[:, :, :, -1] = 0.0
                  self.preturb = [delta * standard_preturb for delta in delta_list]
                  self.preturb = np.concatenate(self.preturb, axis=0) #[delta,224,224,3]

            else:
                  self.preturb = [np.random.normal(loc=0.0, scale= delta, size=[1, self.img_size, self.img_size, 3]) for delta in delta_list]
                  self.preturb = np.concatenate(self.preturb, axis=0)
                  if grating:
                        self.preturb[:, :, :, -1] = 0.0
 
            self.preturb_norm = np.linalg.norm(self.preturb.reshape([len(self.preturb), -1]), axis=-1)

            #################### Clipping ###################################
            #self.noised_inputs = np.clip(self.input_fig + self.preturb, 0, 1) #[delta,224,224,3]
            self.noised_inputs = self.input_fig + self.preturb
            #################### Clipping ###################################

            #统计被clip的数量
            self.images_without_clip=self.input_fig + self.preturb #[delta,224,224,3]

            self.smaller_than_zero=(self.images_without_clip<0)+0 #[delta,224,224,3] #+0将（TRUE，FALSE变为1,0）
            self.smaller_than_zero_vector=self.smaller_than_zero.reshape([len(self.smaller_than_zero), -1]) #[delta,150528]

            self.bigger_than_one=(self.images_without_clip>1)+0 
            self.bigger_than_one_vector=self.bigger_than_one.reshape([len(self.bigger_than_one), -1])

            self.clip_number=np.sum(self.smaller_than_zero_vector,axis=1)+np.sum(self.bigger_than_one_vector,axis=1)
            self.clip_ratio=self.clip_number / (3*self.img_size*self.img_size)
        
      def get_results(self,Norm=None):
            '''
            数据处理，得到需要的指标
            Norm:用什么范数，默认为None(二范数),1:1范数，2：2范数，np.inf：无穷范数
            encoder_num:encoder的层数
            生成三个列表，分别储存embedding、encoder、logit层攻击之前输出的二范数和攻击之后error的二范数
            '''
            self.Norm=Norm
            # 进行攻击 embedding_shape[delta,150528] mid_shape[delta,num of mid ,150528]
            result_list = [(self.model(noised_input[None, :], self.encoder_num)) for noised_input in self.noised_inputs]
            self.attack_embedding = np.concatenate([result[0][None, :] for result in result_list], axis=0)#扰动后embedding层输出
            self.attack_encoder = np.concatenate([result[1][None, :] for result in result_list], axis=0)  #扰动后encoder层输出
            self.attack_logit = np.concatenate([result[2][None, :] for result in result_list], axis=0)    #扰动后logit层输出
            self.error_embedding_vector = self.attack_embedding - self.unattacked_output[0][None, :]      #embedding层误差
            self.error_encoder_vector = self.attack_encoder - self.unattacked_output[1][None, :]          #encoder层误差
            self.error_logit_vector = self.attack_logit - self.unattacked_output[2][None, :]              #logit层误差

            #embedding层：
            self.embedding_list=[]
            self.error_embedding = np.linalg.norm(self.error_embedding_vector,ord=self.Norm, axis=1) # embedding层误差的二范数
            self.embedding = np.linalg.norm(self.unattacked_output[0][None, :],ord=self.Norm, axis=1) # 扰动前
            self.embedding_list.append(self.embedding)
            self.embedding_list.append(self.error_embedding)
            #self.relative_embedding= self.error_embedding /self.embedding # [len,]
            #embedding_list.append(self.relative_embedding)

            #encoder层
            self.encoder_list=[]  #将下面两个list放在一起，encoder_list[0]=encoder_ls,encoder_list[1]=error_encoder_ls
            error_encoder_ls=[]  #各个encoder层的误差的二范数
            encoder_ls=[]  #没有扰动之前各个encoder层的输出的二范数，长度为encoder_num
            for num in range(self.encoder_num):
                  self.error_encoder_single_vector=self.error_encoder_vector[:,num,:]  #第num层误差
                  self.unattacked_single_vector=self.unattacked_output[1][num,:] #第num层扰动之前
                  self.error_encoder_single = np.linalg.norm(self.error_encoder_single_vector,ord=self.Norm, axis=1) # 第num层误差二范数
                  error_encoder_ls.append(self.error_encoder_single)
                  self.encoder = np.linalg.norm(self.unattacked_single_vector[None, :], ord=self.Norm, axis=1) # 扰动前的二范数
                  encoder_ls.append(self.encoder)
            self.encoder_list.append(encoder_ls)
            self.encoder_list.append(error_encoder_ls)
         

            #logit层：
            self.logit_list=[]
            self.logit = np.linalg.norm(self.unattacked_output[2][None, :],ord=self.Norm, axis=1) # 扰动前
            self.error_logit = np.linalg.norm(self.error_logit_vector,ord=self.Norm, axis=1) # logit层误差的二范数
            self.logit_list.append(self.logit)
            self.logit_list.append(self.error_logit)
            # self.relative_logit= self.error_logit/self.logit # [len,]
            # logit_list.append(self.relative_logit)

      #返回所有结果的列表     
      def get_all(self):
            return self.embedding_list, self.encoder_list, self.logit_list

      #返回embedding层的数据：
      def get_embedding(self):
            pass

      #返回encoder层的数据：
      def get_encoder(self):
            pass

      #返回logit层的数据：
      def get_logit(self):
            pass

class Plot():
      def __init__(self,dir,delta_list):

            self.dir=dir #储存路径
            self.delta_list=delta_list #delta取值       
      
      def plot_embedding(self,embedding_list):
            self.error_embedding=embedding_list[1]
            #绘图
            plt.figure(figsize=(8,8))
            ax1 = plt.subplot(111)
            plt.plot(self.delta_list, self.error_embedding,'r'+'o-',label='error_embedding')
            ax1.legend(loc=1)
            plt.xlim((0,0.52))
            plt.ylim((0,np.max(self.error_embedding)+2))
            ax1.set_ylabel('error_embedding');
            ax1.set_xlabel('delta')

            #保存图片
            if not os.path.exists(self.dir):
                  os.mkdir(self.dir)
            plt.savefig(os.path.join(self.dir, 'embedding.png'))#第一个是指存储路径，第二个是图片名字
            plt.close()
  
    
      def plot_encoder(self,encoder_num,encoder_list):
            # num=[0,encoder_num-1]
            self.error_encoder=encoder_list[1]
            #绘图
            plt.figure(figsize=(8,8))
            ax1 = plt.subplot(111)
            for i in range(encoder_num):
                  plt.plot(self.delta_list, self.error_encoder[i],label='error_encoder{}'.format(i+1))
                  ax1.legend(loc=i)
            plt.xlim((0,0.52))
            y_max=max(np.max(self.error_encoder[0]),np.max(self.error_encoder[1]),np.max(self.error_encoder[2]))
            plt.ylim((0,y_max+2))
            ax1.set_ylabel('error_encoder');
            ax1.set_xlabel('delta')

            plt.suptitle('error_encoder')
            #保存图片
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
            plt.savefig(os.path.join(self.dir, 'encoder.png'))#第一个是指存储路径，第二个是图片名字
            plt.close()
            plt.clf() 
    
      def plot_logit(self,logit_list):
            self.error_logit=logit_list[1]
            #绘图
            plt.figure(figsize=(8,8))
            ax1 = plt.subplot(111)
            plt.plot(self.delta_list, self.error_logit,'r'+'o-',label='error_logit')
            ax1.legend(loc=1)
            plt.xlim((0,0.52))
            plt.ylim((0,np.max(self.error_logit)+2))
            ax1.set_ylabel('error_logit');
            ax1.set_xlabel('delta')

            #保存图片
            if not os.path.exists(self.dir):
                  os.mkdir(self.dir)
            plt.savefig(os.path.join(self.dir, 'logit.png'))#第一个是指存储路径，第二个是图片名字
            plt.close()
  

      def show_evolution(self): # dir: 'show_evolution/子文件夹名/'
            if not os.path.exists(self.dir):
                  os.mkdir(self.dir)
            for k in range(len(self.preturb)):
                  fig, ax = plt.subplots(1, 3)
                  fig.set_size_inches(9, 3)
                  ax[0].imshow(self.input_fig[0])# [h, w, 3]
                  ax[0].set_axis_off()
                  ax[1].imshow(0.5 + self.preturb[k])
                  ax[1].set_axis_off()
                  ax[2].imshow(self.noised_inputs[k])
                  ax[2].set_axis_off()
                  plt.savefig(self.dir + f'{k}.jpg')
                  plt.close()