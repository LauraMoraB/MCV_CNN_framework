import sys
import time
import numpy as np
import os

sys.path.append('../')
from metrics.metrics import compute_precision, compute_recall, compute_f1score, compute_accuracy, \
    compute_confusion_matrix, extract_stats_from_confm
from tasks.simple_trainer_manager  import SimpleTrainer
from utils.tools import confm_metrics2image

class Classification_Manager(SimpleTrainer):
    def __init__(self, cf, model):
        super(Classification_Manager, self).__init__(cf, model)

    class train(SimpleTrainer.train):

        def __init__(self, logger_stats, model, cf, validator, stats, msg):
            super(Classification_Manager.train, self).__init__(logger_stats, model, cf, validator, stats, msg)
            if self.cf.resume_experiment:
                self.msg.msg_stats_best = 'Best case: epoch = %d, acc= %.2f, precision= %.2f, recall= %.2f, ' \
                                          'f1score= %.2f, loss = %.5f\n' % (self.model.best_stats.epoch,
                            100 * self.model.best_stats.val.acc, 100 * self.model.best_stats.val.precision,
                            100 * self.model.best_stats.val.recall, 100 * self.model.best_stats.val.f1score,
                            self.model.best_stats.val.loss)

        def validate_epoch(self, valid_set, valid_loader, early_stopping, epoch):
            if valid_set is not None and valid_loader is not None:
                # Set model in validation mode
                self.model.net.eval()

                self.validator.start(valid_set, valid_loader, 'Epoch Validation', epoch)

                # Early stopping checking
                if self.cf.early_stopping:
                    if early_stopping.check(self.stats.train.loss, self.stats.val.loss, self.stats.val.mIoU,
                                            self.stats.val.acc, self.stats.val.f1score):
                        self.stop = True

                # Set model in training mode
                self.model.net.train()

        def compute_stats(self, confm_list, train_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)

            TP = np.sum(TP_list)
            TN = np.sum(TN_list)
            FP = np.sum(FP_list)
            FN = np.sum(FN_list)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            self.stats.train.acc = np.sum(TP_list)/(np.sum(FP_list)+np.sum(TP_list))
            self.stats.train.recall = recall
            self.stats.train.precision = precision

            print("VAL results: accuracy", self.stats.train.acc, "precision", precision, "recall", recall)
            self.stats.train.f1score = 2 * (recall * precision) / (recall + precision)
            if train_loss is not None:
                self.stats.train.loss = train_loss.avg

        def save_stats_epoch(self, epoch):
            # Save logger
            if epoch is not None:
                # Epoch loss tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.train.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.train.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100.*self.stats.train.precision, epoch)
                self.writer.add_scalar('metrics/recall', 100.*self.stats.train.recall, epoch)
                self.writer.add_scalar('metrics/f1score', 100.*self.stats.train.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.train.get_confm_norm(), self.cf.labels)
                # print(self.writer.add_image.__code__.co_varnames)
                # self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')


    class validation(SimpleTrainer.validation):
        def __init__(self, logger_stats, model, cf, stats, msg):
            super(Classification_Manager.validation, self).__init__(logger_stats, model, cf, stats, msg)

        def compute_stats(self, confm_list, val_loss):
            TP_list, TN_list, FP_list, FN_list = extract_stats_from_confm(confm_list)

            TP = np.sum(TP_list)
            TN = np.sum(TN_list)
            FP = np.sum(FP_list)
            FN = np.sum(FN_list)

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            self.stats.val.acc = np.sum(TP_list)/(np.sum(FP_list)+np.sum(TP_list))
            self.stats.val.recall = recall
            self.stats.val.precision = precision

            print("VAL results: accuracy", self.stats.val.acc, "precision", precision, "recall", recall)
            self.stats.val.f1score = 2 * (recall * precision) / (recall + precision)
            if val_loss is not None:
                self.stats.val.loss = val_loss.avg

        def save_stats(self, epoch):
            # Save logger
            if epoch is not None:
                # add scores to log
                self.logger_stats.write('----------------- Epoch scores summary -------------------------\n')
                self.logger_stats.write(
                    '[epoch %d], [val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        epoch, self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')

                # add scores to tensorboard
                self.writer.add_scalar('losses/epoch', self.stats.val.loss, epoch)
                self.writer.add_scalar('metrics/accuracy', 100.*self.stats.val.acc, epoch)
                self.writer.add_scalar('metrics/precision', 100.*self.stats.val.precision, epoch)
                self.writer.add_scalar('metrics/recall', 100.*self.stats.val.recall, epoch)
                self.writer.add_scalar('metrics/f1score', 100.*self.stats.val.f1score, epoch)
                conf_mat_img = confm_metrics2image(self.stats.val.get_confm_norm(), self.cf.labels)
                # self.writer.add_image('metrics/conf_matrix', conf_mat_img, epoch, dataformats='HWC')
            else:
                self.logger_stats.write('----------------- Scores summary --------------------\n')
                self.logger_stats.write(
                    '[val loss %.5f], [acc %.2f], [precision %.2f], [recall %.2f], [f1score %.2f]\n' % (
                        self.stats.val.loss, 100 * self.stats.val.acc, 100 * self.stats.val.precision,
                        100 * self.stats.val.recall, 100 * self.stats.val.f1score))
                self.logger_stats.write('---------------------------------------------------------------- \n')


    class predict(SimpleTrainer.predict):
        def __init__(self, logger_stats, model, cf):
            super(Classification_Manager.predict, self).__init__(logger_stats, model, cf)
            self.filename = os.path.join(self.cf.predict_path_output, 'predictions.txt')
            self.f = open(self.filename,'w')

        def write_results(self, predictions, img_name, img_shape):
            msg = img_name[0] + ' ' + str(predictions[0]) + '\n'
            self.f.writelines(msg)
