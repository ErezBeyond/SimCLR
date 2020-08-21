import torch
from models.resnet_simclr import ResNetSimCLR, get_resnet_goemetric_transform
from utils.coordinate_transforms import affine_transform_boxes
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from downstream_tasks.downstream_tests import test_downstream_classification
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.N_patches = 100
        self.rects_array = self.generate_rects(self.N_patches, (684, 456),
                                               min_h=96, min_w=96)
        # xx, yy = np.meshgrid(np.linspace(0, 1, self.N_patches+1), np.linspace(0, 1, self.N_patches+1))
        # x_mins = xx[0:-1,0:-1].reshape(-1,1)
        # x_maxs = xx[0:-1, 1:].reshape(-1, 1)
        # y_mins = yy[0:-1, 0:-1].reshape(-1, 1)
        # y_maxs = yy[1:, 0:-1].reshape(-1, 1)
        # self.rects_array = np.hstack((x_mins, y_mins, x_maxs, y_maxs))
        self.max_N_patches = 300#int(config['batch_size']*self.N_patches**2 / 2)
        self.nt_xent_criterion_patches = NTXentLoss(self.device, self.max_N_patches, **config['loss'])

    def generate_rects(self, num_rects, image_size, min_h=100, min_w=100):
        x_mins = np.random.random_integers(0, image_size[1]-min_w-1, (num_rects,1))
        y_mins = np.random.random_integers(0, image_size[0] - min_h-1, (num_rects,1))
        x_maxs = np.random.randint(x_mins+min_w, image_size[1])
        y_maxs = np.random.randint(y_mins + min_h, image_size[0])
        return np.hstack((x_mins/image_size[1], y_mins/image_size[0], x_maxs/image_size[1], y_maxs/image_size[0]))

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis, _ = model(xis['img'])  # [N,C]

        # get the representations and the projections
        rjs, zjs, _ = model(xjs['img'])  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def global_roi_pooling(self, feature_map, rois):
        """
        :param feature_map: (C, H, W)
        :param rois: (N, 4) N refers to bbox num, 4 represent (x0, y0, x1, y1)
        :param size: output size
        :return: (1, C, size[0], size[1])
        """
        output = []
        for roi in rois:
            output.append(F.adaptive_avg_pool2d(feature_map[:, roi[1]:roi[3], roi[0]:roi[2]], (1, 1)))

        return torch.stack(output)

    @staticmethod
    def get_good_cooresponding_indices(rois_i, rois_j, shape):
        good_ind_i = np.logical_and.reduce((rois_i[:, 0] <= shape[1],
                                          rois_i[:, 0] >= 0,
                                          rois_i[:, 2] <= shape[1],
                                          rois_i[:, 2] >= 0,
                                          rois_i[:, 1] <= shape[0],
                                          rois_i[:, 1] >= 0,
                                          rois_i[:, 3] <= shape[0],
                                          rois_i[:, 3] >= 0), axis=0)

        good_ind_j = np.logical_and.reduce((rois_j[:, 0] <= shape[1],
                                            rois_j[:, 0] >= 0,
                                            rois_j[:, 2] <= shape[1],
                                            rois_j[:, 2] >= 0,
                                            rois_j[:, 1] <= shape[0],
                                            rois_j[:, 1] >= 0,
                                            rois_j[:, 3] <= shape[0],
                                            rois_j[:, 3] >= 0), axis=0)
        return np.logical_and(good_ind_i, good_ind_j)

    def matched_roi_pooling(self, cis, cjs, His, Hjs, H_model, img_size):
        p_is = []
        p_js = []
        for ci, cj, Hi, Hj in zip(cis, cjs, His.numpy(), Hjs.numpy()):
            canonic_rects = self.rects_array * [img_size[1], img_size[0], img_size[1], img_size[0]]
            rois_i = affine_transform_boxes(canonic_rects, H_model @ Hi).astype(np.int)
            rois_j = affine_transform_boxes(canonic_rects, H_model @ Hj).astype(np.int)
            good_ind = SimCLR.get_good_cooresponding_indices(rois_i, rois_j, cis.shape[2:])
            p_is.append(self.global_roi_pooling(ci, rois_i[good_ind, :]))
            p_js.append(self.global_roi_pooling(cj, rois_j[good_ind, :]))
        p_is = torch.cat(p_is)
        p_js = torch.cat(p_js)
        return p_is[0:self.max_N_patches], p_js[0:self.max_N_patches]

    def _visualize(self, xis, xjs, img_idx=0, k=0):

        img_size = xis['img'].shape[2:]
        canonic_rects = self.rects_array * [img_size[1], img_size[0], img_size[1], img_size[0]]
        rois_i = affine_transform_boxes(canonic_rects, xis['H'][img_idx].numpy()).astype(np.int)
        rois_j = affine_transform_boxes(canonic_rects, xjs['H'][img_idx].numpy()).astype(np.int)
        I1 = (255 * xis['img'][0].cpu().numpy()).astype(np.uint8).transpose(1, 2, 0)
        I2 = (255 * xjs['img'][0].cpu().numpy()).astype(np.uint8).transpose(1, 2, 0)
        from matplotlib import pyplot as plt
        rois_i = rois_i.astype(np.int)
        rois_j = rois_j.astype(np.int)
        plt.imshow(I1[rois_i[k][1]:rois_i[k][3], rois_i[k][0]:rois_i[k][2]])
        plt.show()
        plt.imshow(I2[rois_j[k][1]:rois_j[k][3], rois_j[k][0]:rois_j[k][2]])
        plt.show()
        from utils.visualizations import imshow_and_boxes
        imshow_and_boxes(I1, rois_i[k:(k + 1)])
        imshow_and_boxes(I2, rois_j[k:(k + 1)])
        import cv2
        II1 = cv2.warpAffine(I1, np.linalg.inv(xis['H'][img_idx].numpy())[0:2], dsize=(I1.shape[1], I1.shape[0]))
        II2 = cv2.warpAffine(I2, np.linalg.inv(xjs['H'][img_idx].numpy())[0:2], dsize=(I1.shape[1], I1.shape[0]))
        plt.imshow(II1)
        plt.show()
        plt.imshow(II2)
        plt.show()
        imshow_and_boxes(II1, canonic_rects[k:(k + 1)])
        imshow_and_boxes(II2, canonic_rects[k:(k + 1)])



    def _step_with_patches(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis, cis = model(xis['img'])  # [N,C]
        # get the representations and the projections
        rjs, zjs, cjs = model(xjs['img'])  # [N,C]
        H_model = get_resnet_goemetric_transform()
        self.rects_array = self.generate_rects(self.N_patches, (684, 456),
                                               min_h=64, min_w=64)
        p_is, p_js = self.matched_roi_pooling(cis, cjs, xis['H'], xjs['H'], H_model, img_size=xis['img'].shape[2:])
        # use this to visualize patches
        # self._visualize(xis, xjs, 0 , 5)

        zis_p = model.project(p_is.squeeze())
        zjs_p = model.project(p_js.squeeze())
        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        zis_p = F.normalize(zis_p, dim=1)
        zjs_p = F.normalize(zjs_p, dim=1)

        loss_fullframe = self.nt_xent_criterion(zis, zjs)
        loss_patches = self.nt_xent_criterion_patches(zis_p, zjs_p)
        loss = loss_fullframe + loss_patches
        return loss

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):

        # train_loader, valid_loader = self.dataset.get_data_loaders()
        train_loader, valid_loader = self.dataset.get_rockwell_data_loaders()


        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-5, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis['img'] = xis['img'].to(self.device)
                xjs['img'] = xjs['img'].to(self.device)
                # xis['H'] = xis['H'].to(self.device)
                # xjs['H'] = xjs['H'].to(self.device)

                loss = self._step_with_patches(model, xis, xjs, n_iter)
                #loss = self._step(model, xis, xjs, n_iter)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
                test_acc = self._test(model)
                self.writer.add_scalar('test_acc', test_acc, global_step=valid_n_iter)


            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            print("Loaded pre-trained model with success.")
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for counter, ((xis, xjs), _) in enumerate(valid_loader):
                xis['img'] = xis['img'].to(self.device)
                xjs['img'] = xjs['img'].to(self.device)
                # loss = self._step_with_patches(model, xis, xjs, counter)
                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
            valid_loss /= (counter+1)
        model.train_classifier()
        return valid_loss

    def _test(self, model):
        model.eval()
        test_acc = test_downstream_classification(model.features, out_dim=2048)
        model.train_classifier()
        return test_acc