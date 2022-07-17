import torch
import torch.utils.data
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import numpy as np
import segmentation_models_pytorch as smp
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
import copy


class LightModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        aux_params = dict(pooling='avg',
                          dropout=0.5,
                          activation=None,
                          classes=5)  # 0-4
        self.model = smp.Unet(encoder_name='resnet18',
                              encoder_weights="imagenet",
                              in_channels=3,
                              activation=None,
                              classes=4,
                              aux_params=aux_params)
        self.train_iou = smp.utils.metrics.IoU(threshold=0.5)
        self.valid_iou = smp.utils.metrics.IoU(threshold=0.5)
        self.diceLoss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')
        self.train_ce = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy(num_classes=5, average=None)
        self.test_auprc_EX = torchmetrics.AveragePrecision(pos_label=1)
        self.test_auprc_HE = torchmetrics.AveragePrecision(pos_label=1)
        self.test_auprc_MA = torchmetrics.AveragePrecision(pos_label=1)
        self.test_auprc_SE = torchmetrics.AveragePrecision(pos_label=1)
        self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.sum_test_iou = 0
        self.sum_test_dice = 0
        self.train_grades = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.valid_grades = {0: [], 1: [], 2: [], 3: [], 4: []}
        self.true_test1 = []
        self.pred_test1 = [0, 0, 0, 0, 0]  # classes: 0-4
        self.true_test2 = []
        self.pred_test2 = [0, 0, 0, 0, 0]  # classes: 1-4
        self.test_y_pixel = []
        self.test_y_hat_pixel = []
        self.test_y_EX = []
        self.test_y_HE = []
        self.test_y_MA = []
        self.test_y_SE = []
        self.test_p_EX = []
        self.test_p_HE = []
        self.test_p_MA = []
        self.test_p_SE = []

    def forward(self, x):
        x, label = self.model(x)
        return x, label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1,
                                                            last_epoch=-1)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
        #                                                          patience=3, threshold=0.0001)
        scheduler = {'scheduler': lr_scheduler,
                     'interval': 'epoch',
                     'name': 'Adam lr',
                     'frequency': 1,
                     'monitor': 'loss'}
        return [optimizer]  # , [scheduler]

    def sum_to_grades_dict(self, d_grades, grades):
        grades = grades.cpu().numpy().tolist()
        for i in range(0, 5):
            d_grades[i].append(grades.count(i))
        return d_grades

    def training_step(self, batch, batch_idx):
        x1, dr_grade1, x2, y2, dr_grade2 = batch[0][0], batch[0][1], batch[1][0], batch[1][1], batch[1][2]
        _, label1 = self(x1)
        y_hat2, label2 = self(x2)
        # self.train_grades = self.sum_to_grades_dict(self.train_grades, dr_grade)
        ce_segm = self.train_ce(label2, dr_grade2)  # CE applies softmax, true must not be one hot
        ce_class = self.train_ce(label1, dr_grade1)
        # Accuracy - CE
        self.log('train_acc', self.train_acc(label1, dr_grade1), on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_ce_segm', ce_segm, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_ce_class', ce_class, on_step=False, on_epoch=True, prog_bar=False)
        # BCE
        train_bce = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).sum(dim=1).mean()
        bce_per_lesion = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).mean(dim=0)  # size=4
        self.log('train_bce_EX', bce_per_lesion[0], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_bce_HE', bce_per_lesion[1], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_bce_MA', bce_per_lesion[2], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_bce_SE', bce_per_lesion[3], on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_bce', train_bce, on_step=False, on_epoch=True, prog_bar=True)
        # DICE
        train_dice = self.diceLoss(y_hat2, y2)
        self.log('train_dice', train_dice, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_dice_EX', self.diceLoss(y_hat2[:, 0, :, :], y2[:, 0, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_dice_HE', self.diceLoss(y_hat2[:, 1, :, :], y2[:, 1, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_dice_MA', self.diceLoss(y_hat2[:, 2, :, :], y2[:, 2, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_dice_SE', self.diceLoss(y_hat2[:, 3, :, :], y2[:, 3, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        # IoU
        y_hat_probs2 = self.sigmoid(y_hat2)
        self.log('train_IoU', self.train_iou(y_hat_probs2, y2), on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_IoU_EX', self.train_iou(y_hat_probs2[:, 0, :, :], y2[:, 0, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_IoU_HE', self.train_iou(y_hat_probs2[:, 1, :, :], y2[:, 1, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_IoU_MA', self.train_iou(y_hat_probs2[:, 2, :, :], y2[:, 2, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('train_IoU_SE', self.train_iou(y_hat_probs2[:, 3, :, :], y2[:, 3, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        return train_bce + ce_segm + ce_class

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x1, dr_grade1, x2, y2, dr_grade2 = batch[0][0], batch[0][1], batch[1][0], batch[1][1], batch[1][2]
        _, label1 = self(x1)
        y_hat2, label2 = self(x2)
        # self.valid_grades = self.sum_to_grades_dict(self.valid_grades, dr_grade)
        ce_segm = self.train_ce(label2, dr_grade2)  # CE applies softmax, true must not be one hot
        ce_class = self.train_ce(label1, dr_grade1)
        # Accuracy
        self.log('valid_acc', self.valid_acc(label1, dr_grade1), on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_ce_segm', ce_segm, on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_ce_class', ce_class, on_step=False, on_epoch=True, prog_bar=False)
        # BCE
        valid_bce = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).sum(dim=1).mean()
        bce_per_lesion = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).mean(dim=0)  # size=4
        self.log('valid_bce', valid_bce, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_bce_EX', bce_per_lesion[0], on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_bce_HE', bce_per_lesion[1], on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_bce_MA', bce_per_lesion[2], on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_bce_SE', bce_per_lesion[3], on_step=False, on_epoch=True, prog_bar=False)
        # DICE
        valid_dice = self.diceLoss(y_hat2, y2)
        self.log('valid_dice', valid_dice, on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_dice_EX', self.diceLoss(y_hat2[:, 0, :, :], y2[:, 0, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_dice_HE', self.diceLoss(y_hat2[:, 1, :, :], y2[:, 1, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_dice_MA', self.diceLoss(y_hat2[:, 2, :, :], y2[:, 2, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_dice_SE', self.diceLoss(y_hat2[:, 3, :, :], y2[:, 3, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        # IoU
        y_hat_probs2 = self.sigmoid(y_hat2)
        self.log('valid_IoU', self.valid_iou(y_hat_probs2, y2), on_step=False, on_epoch=True, prog_bar=False)
        self.log('valid_IoU_EX', self.valid_iou(y_hat_probs2[:, 0, :, :], y2[:, 0, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_IoU_HE', self.valid_iou(y_hat_probs2[:, 1, :, :], y2[:, 1, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_IoU_MA', self.valid_iou(y_hat_probs2[:, 2, :, :], y2[:, 2, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        self.log('valid_IoU_SE', self.valid_iou(y_hat_probs2[:, 3, :, :], y2[:, 3, :, :]), on_step=False, on_epoch=True,
                 prog_bar=False)
        loss = valid_bce + ce_class + ce_segm
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        x1, dr_grade1, x2, y2, dr_grade2 = batch[0][0], batch[0][1], batch[1][0], batch[1][1], batch[1][2]
        _, label1 = self(x1)

        probs1 = self.soft_max(label1)
        true1 = dr_grade1.cpu().detach().item()
        pred1 = probs1.cpu().detach().numpy()
        self.true_test1.append(true1)
        self.pred_test1 = np.row_stack((self.pred_test1, pred1)).tolist()

        loss = self.train_ce(label1, dr_grade1)
        self.log('test_acc', self.valid_acc(label1, dr_grade1), on_step=True, on_epoch=False, prog_bar=False)
        self.log('test_ce_class', loss, on_step=True, on_epoch=False, prog_bar=False)

        if batch_idx < 225:  # batch_idx < LENGTH of the segmentation test dataset (smallest)
            y_hat2, label2 = self(x2)

            probs2 = self.soft_max(label2)
            true2 = dr_grade2.cpu().detach().item()
            pred2 = probs2.cpu().detach().numpy()
            self.true_test2.append(true2)
            self.pred_test2 = np.row_stack((self.pred_test2, pred2)).tolist()

            y_hat2_probs = self.sigmoid(y_hat2)
            self.test_y_EX.append(y2[:, 0, :, :].detach().cpu().flatten().numpy().astype('int8'))
            self.test_y_HE.append(y2[:, 1, :, :].detach().cpu().flatten().numpy().astype('int8'))
            self.test_y_MA.append(y2[:, 2, :, :].detach().cpu().flatten().numpy().astype('int8'))
            self.test_y_SE.append(y2[:, 3, :, :].detach().cpu().flatten().numpy().astype('int8'))
            self.test_p_EX.append(y_hat2_probs[:, 0, :, :].detach().cpu().flatten().numpy().astype('float16'))
            self.test_p_HE.append(y_hat2_probs[:, 1, :, :].detach().cpu().flatten().numpy().astype('float16'))
            self.test_p_MA.append(y_hat2_probs[:, 2, :, :].detach().cpu().flatten().numpy().astype('float16'))
            self.test_p_SE.append(y_hat2_probs[:, 3, :, :].detach().cpu().flatten().numpy().astype('float16'))
            # Accuracy

            self.log('test_ce_segm', self.train_ce(label2, dr_grade2), on_step=True, on_epoch=False, prog_bar=False)

            # BCE
            test_bce = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).sum(dim=1).mean()
            bce_per_lesion = self.bceloss(y_hat2, y2).mean(dim=(2, 3)).mean(dim=0)  # size=4
            self.log('test_bce', test_bce, on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_bce_EX', bce_per_lesion[0], on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_bce_HE', bce_per_lesion[1], on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_bce_MA', bce_per_lesion[2], on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_bce_SE', bce_per_lesion[3], on_step=True, on_epoch=False, prog_bar=False)
            # DICE
            test_dice = self.diceLoss(y_hat2, y2)
            self.sum_test_dice += test_dice
            self.log('test_dice', test_dice, on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_dice_EX', self.diceLoss(y_hat2[:, 0, :, :], y2[:, 0, :, :]), on_step=True, on_epoch=False,
                     prog_bar=False)
            self.log('test_dice_HE', self.diceLoss(y_hat2[:, 1, :, :], y2[:, 1, :, :]), on_step=True, on_epoch=False,
                     prog_bar=False)
            self.log('test_dice_MA', self.diceLoss(y_hat2[:, 2, :, :], y2[:, 2, :, :]), on_step=True, on_epoch=False,
                     prog_bar=False)
            self.log('test_dice_SE', self.diceLoss(y_hat2[:, 3, :, :], y2[:, 3, :, :]), on_step=True, on_epoch=False,
                     prog_bar=False)
            # IoU
            self.sum_test_iou += self.valid_iou(y_hat2_probs, y2)
            self.log('test_IoU', self.valid_iou(y_hat2_probs, y2), on_step=True, on_epoch=False, prog_bar=False)
            self.log('test_IoU_EX', self.valid_iou(y_hat2_probs[:, 0, :, :], y2[:, 0, :, :]), on_step=True,
                     on_epoch=False,
                     prog_bar=False)
            self.log('test_IoU_HE', self.valid_iou(y_hat2_probs[:, 1, :, :], y2[:, 1, :, :]), on_step=True,
                     on_epoch=False,
                     prog_bar=False)
            self.log('test_IoU_MA', self.valid_iou(y_hat2_probs[:, 2, :, :], y2[:, 2, :, :]), on_step=True,
                     on_epoch=False,
                     prog_bar=False)
            self.log('test_IoU_SE', self.valid_iou(y_hat2_probs[:, 3, :, :], y2[:, 3, :, :]), on_step=True,
                     on_epoch=False,
                     prog_bar=False)
        return loss  # test_bce

    def test_epoch_end(self, outputs):
        # AUPRC at pixels
        self.test_y_EX = np.concatenate(self.test_y_EX)
        self.test_y_HE = np.concatenate(self.test_y_HE)
        self.test_y_MA = np.concatenate(self.test_y_MA)
        self.test_y_SE = np.concatenate(self.test_y_SE)
        self.test_p_EX = np.concatenate(self.test_p_EX)
        self.test_p_HE = np.concatenate(self.test_p_HE)
        self.test_p_MA = np.concatenate(self.test_p_MA)
        self.test_p_SE = np.concatenate(self.test_p_SE)
        scoreEx = skm.average_precision_score(self.test_y_EX, self.test_p_EX, pos_label=1)
        scoreHe = skm.average_precision_score(self.test_y_HE, self.test_p_HE, pos_label=1)
        scoreMa = skm.average_precision_score(self.test_y_MA, self.test_p_MA, pos_label=1)
        scoreSe = skm.average_precision_score(self.test_y_SE, self.test_p_SE)
        print('Total auprc EX: ' + str(scoreEx))
        print('Total auprc HE: ' + str(scoreHe))
        print('Total auprc MA: ' + str(scoreMa))
        print('Total auprc SE: ' + str(scoreSe))

        # ROC score for multiclass classification
        self.pred_test1 = self.pred_test1[1:]
        score = skm.roc_auc_score(self.true_test1, self.pred_test1, multi_class='ovr')
        self.log('ROC AUC score for test data, classes:0-4', score, on_step=False, on_epoch=True, prog_bar=False)
        #
        # Cohen Kappa score
        pred_labels = np.argmax(self.pred_test1, axis=1)
        cohen = skm.cohen_kappa_score(self.true_test1, pred_labels)
        print('Kappa 0-4: ' + str(cohen))
        # Accuracy
        acc = skm.accuracy_score(self.true_test1, pred_labels)
        print('Overall Accuracy 0-4: ' + str(acc))
        true_tensor = torch.tensor(self.true_test1, device='cuda')
        pred_tensor = torch.tensor(self.pred_test1, device='cuda')
        acc_classes = self.test_acc(pred_tensor, true_tensor)
        print('Accuracy for 5 classes:')
        print(acc_classes)
        # Confusion Matrix
        conf = skm.confusion_matrix(self.true_test1, pred_labels)
        print('Confusion Matrix')
        print(conf)
        #
        y_true_binarize1 = np.array(label_binarize(self.true_test1, classes=[0, 1, 2, 3, 4]))
        y_pred_arr1 = np.array(self.pred_test1)
        # ROC curve for multiclass classification
        """
        fig = plt.figure(figsize=(16, 5))
        plt.subplot(1, 1, 1)
        fpr = dict()
        tpr = dict()
        print('AUROC score for classes 0-4:')
        for i in range(5):
            fpr[i], tpr[i], _ = skm.roc_curve(y_true_binarize1[:, i],
                                          y_pred_arr1[:, i])
            plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
            print(skm.roc_auc_score(y_true_binarize1[:, i], y_pred_arr1[:, i]))
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title("ROC curve")
        #plt.show()
        neptune.log_image('roc', fig)  # update to neptune new
        """
        #
        # ROC score for multiclass classification FOR MULTI TASK
        self.pred_test2 = self.pred_test2[1:]
        try:
            score = skm.roc_auc_score(self.true_test2, self.pred_test2, multi_class='ovr')
            self.log('ROC AUC score for test data, classes:1-4', score, on_step=False, on_epoch=True, prog_bar=False)
        except:
            print("Can't process total roc for 1-4 classes")
            pass
        # Cohen Kappa Score
        pred_labels = np.argmax(self.pred_test2, axis=1)
        cohen = skm.cohen_kappa_score(self.true_test2, pred_labels)
        print('Kappa 1-4: ' + str(cohen))
        # Accuracy
        acc = skm.accuracy_score(self.true_test2, pred_labels)
        print('Overall Accuracy 1-4: ' + str(acc))
        true_tensor = torch.tensor(self.true_test2, device='cuda')
        pred_tensor = torch.tensor(self.pred_test2, device='cuda')
        acc_classes = self.test_acc(pred_tensor, true_tensor)
        print('Accuracy for 4 classes:')
        print(acc_classes)
        #
        y_true_binarize2 = np.array(label_binarize(self.true_test2, classes=[0, 1, 2, 3, 4]))
        y_pred_arr2 = np.array(self.pred_test2)
        # ROC curve for multiclass classification
        """
        fig = plt.figure(figsize=(16, 5))
        plt.subplot(1, 1, 1)
        fpr = dict()
        tpr = dict()
        print('AUROC for classes 1-4:')
        for i in range(5):
            if i!=0:
                fpr[i], tpr[i], _ = skm.roc_curve(y_true_binarize2[:, i],
                                              y_pred_arr2[:, i])
                plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
                print(skm.roc_auc_score(y_true_binarize2[:, i], y_pred_arr2[:, i]))
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title("ROC curve")
        #plt.show()
        neptune.log_image('roc', fig)
        """
        # precision - recall
        """
        fig = plt.figure(figsize=(16, 5))
        plt.subplot(1, 1, 1)
        precision = dict()
        recall = dict()
        for i in range(5):
            precision[i], recall[i], _ = skm.precision_recall_curve(y_true_binarize1[:,i],
                                                                y_pred_arr1[:,i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
            print(skm.average_precision_score(y_true_binarize1[:,i],y_pred_arr1[:,i]))

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")
        #plt.show()
        neptune.log_image('prc', fig)
        #plt.show()
        """
        pass

    def predict(self, batch, batch_idx):
        x, y = batch
        gt = copy.deepcopy(y).astype('float32')
        image_vis = x.transpose(1, 2, 0)
        image_vis = (image_vis / 2) + 0.5
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        x_tensor = x_tensor.float()
        y_hat, grade_pred = self(x_tensor)
        y = torch.from_numpy(y).unsqueeze(0)

        # BCE
        pred_bce = self.bceloss(y_hat, y).mean(dim=(2, 3)).sum(dim=1).mean()
        bce_per_lesion = self.bceloss(y_hat, y).mean(dim=(2, 3)).mean(dim=0)
        y_hat = self.sigmoid(y_hat)
        auprc = 0
        y_hat = y_hat.detach().squeeze().numpy().round().astype('float32')
        return pred_bce, auprc, y_hat, grade_pred, gt, image_vis.astype('float32')

    def testing_iou_dice(self):
        return self.sum_test_iou, self.sum_test_dice