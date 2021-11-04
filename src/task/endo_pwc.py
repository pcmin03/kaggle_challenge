import numpy as np
import copy
import torch
import torch.nn as nn
from tqdm import tqdm 
from accelerate import Accelerator
from ..metric import MetricTracker
import torch.nn.functional as F
class Endoseg:
    def __init__(self, conf, model,optim,scheduler,
                    log,loss,data,saver,callback):
        self.conf = copy.deepcopy(conf)

        # torch.cuda.device_count() > 1
        self.accelerator = Accelerator(
            device_placement = True,
            split_batches = False,
            fp16 = self.conf.base.use_amp == True,
            cpu = False,
            deepspeed_plugin = None,
            rng_types = None,
            kwargs_handlers = None
        )
        
        self.model,self.optim = self.accelerator.prepare(model,optim)
        self.train_dl,self.valid_dl = self.accelerator.prepare(data['train'],data['valid'])
        # self.train_dl, self.valid_dl = data['train'],data['valid']
        self.criterion = self.accelerator.prepare(loss)
        self.device    = self.accelerator.device
        self.sched     = scheduler
        self.saver     = saver
        self.log       = log
        self.saver     = saver
        self.disable_tqdm = not self.accelerator.is_local_main_process
        self.sigmoid = nn.Softmax(dim=1)
        self.train_metric  = MetricTracker(2)
        self.valid_metric  = MetricTracker(2)
        self.callback = callback
        self.avg = 'macro'
        self.target_metric = 'recall'
        # self.criterion_2 = nn.CrossEntropyLoss(self.conf.loss['ignore_index'])

    @property
    def current_epoch(self):
        return self._current_epoch    

    @property 
    def current_step(self):
        return self._current_step

    def setting_pbar(self,epoch,dl,mode):
        pbar = tqdm(
                    enumerate(dl), 
                    bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
                    total=len(dl), 
                    desc=f"{mode}:{epoch}/{self.conf.hyperparameter.epochs}",
                    disable=self.disable_tqdm)
        return pbar

    def share_network(self,image,label,model,pbar):
        image = image.to(self.device, non_blocking=True).float()
        label = label.to(self.device, non_blocking=True).float()
        
        predic = model(image)
        
        if self.conf.loss['type'] == 'MVL': 
            print(predic.shape)
            mean_loss, variance_loss = self.criterion(predic.view, label)
            loss = mean_loss + variance_loss
            loss += self.criterion_2(predic,label.long())
            # predic = torch.argmax(predic,axis=1)
        elif self.conf.loss['type'] == 'MSE': 
            # print( F.one_hot(label.long(),3).shape)
            loss = self.criterion(predic, F.one_hot(label.long(),3).permute(0,3,1,2).float())
        elif self.conf.loss['type'] == 'bce':
            loss = self.criterion(predic,label)
        elif self.conf.loss['type'] == 'ce':
            predic = self.sigmoid(predic)
            loss = self.criterion(predic, label.long())
            # predic = torch.argmax(predic,axis=1)
            
        else :
            assert "no loss define"
        predic = self.accelerator.gather(predic).detach().cpu().numpy()
        label = self.accelerator.gather(label).detach().cpu().numpy()

        # if self.current_epoch % 10 == 0:
        pbar.set_postfix({'Loss':np.round(loss.item(),2)}) 

        return loss,predic,label

    def train_one_epoch(self, model, dl):
        
        model.train()
        self.train_metric.reset()
        pbar = self.setting_pbar(self.current_epoch,dl,'train')
        
        for step, (image, label) in pbar:
            self._current_step = self.current_epoch*len(dl)+step

            loss,predic,label = self.share_network(image,label,model,pbar)
            self.train_metric.update(label,predic)

            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.optim.step()
            # matric = self.train_metric.seg_metric(self.avg)

        if self.current_epoch % 10 ==0:
            # for i in np.arange(0,1,0.1):
            self.train_metric.output_dic['predic'] = np.max(np.array(self.train_metric.output_dic['predic']),axis=1)
            print(self.train_metric.output_dic['predic'].shape,self.train_metric.output_dic['predic'].max())
            # for i in np.arange(0,1,0.):
        
            matric = self.train_metric.seg_metric(self.avg,0.5)
            print(f'train:{matric}') 
            # for num,matric in enumerate(matrics):
            # matric.update({'optimizer':self.optim})

            self.log.update_log(matric,self.current_epoch,'train') # update self.log step
            self.log.update_histogram(model,self.current_epoch,'train') # update weight histogram 
            # if i == 0: 
            self.log.update_image(image[0],self.current_epoch,'train','img') # update transpose image
            self.log.update_image(label[0],self.current_epoch,'train','label') # update transpose image
            print(predic.shape)
            self.log.update_image(predic[0,1],self.current_epoch,'train',f'predic_pos') # update transpose image
            self.log.update_image(predic[0,2],self.current_epoch,'train',f'predic_neg') # update transpose image
        
        else : 
            matric = 0 
        return loss / len(dl), matric


    @torch.no_grad()
    def eval(self, model, dl):
        
        model.eval()
        pbar = self.setting_pbar(self.current_epoch,dl,'valid')
        self.valid_metric.reset()
        for step, (image, label) in pbar:
            loss,predic,label = self.share_network(image,label,model,pbar)   
            self.valid_metric.update(label,predic)

        if self.current_epoch % 10 ==0 :
            self.valid_metric.output_dic['predic'] = np.max(np.array(self.valid_metric.output_dic['predic']),axis=1)
            print(self.valid_metric.output_dic['predic'].shape)
            # for i in np.arange(0,1,0.3): 

            matric = self.valid_metric.seg_metric(self.avg,0.5) 
            print(f'valid:{matric}')
            self.log.update_log(matric,self.current_epoch,'valid') # update 
            # if i == 0: 
            self.log.update_image(image[0],self.current_epoch,'valid','img') # update transpose image
            self.log.update_image(label[0],self.current_epoch,'valid','label') # update transpose image
            print(predic.shape)
            self.log.update_image(predic[0,1],self.current_epoch,'valid',f'predic_pos') # update transpose image
            self.log.update_image(predic[0,2],self.current_epoch,'valid',f'predic_neg') # update transpose image
        else: 
            matric = 0 

        return loss / len(dl),matric

    def run(self):
        # add graph to tensorboard
        if self.log is not None:
            self.log.update_graph(self.model,None)
        
        #just testing 
        # self._current_epoch = 0 
        # _,_ = self.eval(self.model, self.valid_dl)
        
        for epoch in range(1, self.conf.hyperparameter.epochs + 1):
            self._current_epoch = epoch

            # train
            train_loss, train_metric = self.train_one_epoch(self.model, self.train_dl)
            
            if self.sched != None: 
                self.sched.step(self.current_epoch)

            # eval
            if self.current_epoch % 10 ==0:

                valid_loss, valid_metric = self.eval(self.model, self.valid_dl)
            
                self.accelerator.wait_for_everyone()
                sd = {'checkpoint' : self.accelerator.unwrap_model(self.model).state_dict(),
                    'metric':valid_metric[self.target_metric]}

                self.saver.step(valid_metric[self.target_metric],sd,epoch)
            # self.callback['elary_stop'](valid_metric[self.target_metric],sd)
            
            # print(f'Epoch {self.current_epoch}/{self.conf.hyperparameter.epochs} , train_Loss: {train_loss:.3f}, valid_Loss: {valid_loss:.3f}')
            # self.disable_tqdm
            # self.accelerator.wait_for_everyone()



    # @torch.no_grad()
    # def test(self, model, dl,testing=False):
        
    #     model.eval()
    #     pbar = self.setting_pbar(self.current_epoch,dl,'valid')
        
    #     for step, (image, label) in pbar:
            
    #         loss,predic,label = self.share_network(image,label,model,pbar)
            
    #     labellist = np.array(list(itertools.chain(*self.labellist)))
    #     prediclist = np.array(list(itertools.chain(*self.prediclist)))

        
    #     valmetric = self.cal_metric(loss,prediclist,labellist,self.log,'valid')
    #     self.log.update_image(image,self.current_epoch,'valid') # update transpose image
    #     if testing == True: 
    #         return 0
            

    #     return loss / len(dl), valmetric
