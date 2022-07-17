import pytorch_lightning as pl
from model import LightModel
from DataModule import DDRDataModule
import neptune.new as neptune
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from sacred import Experiment
from neptune.new.integrations.sacred import NeptuneObserver
from data import DDR
from saveMasks import save_preds

ex = Experiment("Exp")
neptune_run = neptune.init(project='', api_token='API_TOKEN')
ex.observers.append(NeptuneObserver(run=neptune_run))
neptune_logger = NeptuneLogger(api_key='API_TOKEN', project="")


if __name__ == '__main__':
    epochs = 100
    batch_size = 16
    segm_data_dir = ""
    segm_gt_dir = ""
    grade_dir = ""
    class_data_dir = ""
    data = DDRDataModule(segm_data_dir, segm_gt_dir, class_data_dir, grade_dir, batch_size=batch_size)

    model = LightModel()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="valid_bce", mode="min",
                                          dirpath="")

    trainer = pl.Trainer(gpus=1, max_epochs=epochs, log_every_n_steps=1,
                         check_val_every_n_epoch=1,
                         default_root_dir="",
                         logger=neptune_logger,
                         callbacks=[lr_monitor, checkpoint_callback],
                         precision=16)  # progress_bar_refresh_rate=2

    trainer.fit(model, data)
    grades_train, grades_valid = data.get_grades_train_valid()
    print('Grades picked for clf:')
    print(grades_train)
    print('Grades picked for seg:')
    print(grades_valid)
    print(checkpoint_callback.best_model_path)  # prints path to the best model's checkpoint
    print(checkpoint_callback.best_model_score) # and prints it score
    best_model = LightModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    #trainer.save_checkpoint("last_model.ckpt")
    #last_model = LightModel.load_from_checkpoint(checkpoint_path='./last_model.ckpt')

    data = DDRDataModule(segm_data_dir, segm_gt_dir, class_data_dir, grade_dir, batch_size=1)
    trainer.test(best_model, data)  # or last_model
    # Save M.0.
    test_iou, test_dice = model.testing_iou_dice()
    length = data.test_dataset.__len__()
    print('IoU mo: ' + str(test_iou / length) + '. Dice mo: ' + str(test_dice / length))

    # Predict Segmentation masks
    pred_dir = ""
    gt_dir = ""
    save_dir = ""
    pred_data = DDR(pred_dir, gt_dir, None, augmentation=None)
    for n in range(len(pred_data)):
        # n = np.random.choice(len(pred_data))
        loss, auprc, pred, grade, gt, x = best_model.predict(pred_data.__getitem__(n), 0)
        # print(loss)
        # print(grade)
        save_preds(pred_data.__getname__(n), gt, pred,save_dir)
