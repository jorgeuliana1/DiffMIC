import logging
import time
import gc

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.stats import ttest_rel
from tqdm import tqdm
from ema import EMA
from model import *

from pretraining.dcg import DCG as AuxCls
from pretraining.resnet import ResNet18
from utils import *
from diffusion_utils import *
from tqdm import tqdm
plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.test_num_timesteps = config.diffusion.test_timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        # initial prediction model as guided condition
        if config.diffusion.apply_aux_cls:
            self.cond_pred_model = AuxCls(config).to(self.device)
            
            if hasattr(config.data, "labels_balance"):
                labels_balance = np.asarray(config.data.labels_balance)
                labels_weight = 1 / labels_balance
                normalized_weight = labels_weight / labels_weight.sum()
                weight_tensor = torch.tensor(normalized_weight, dtype=torch.float).to(self.device)
                self.aux_cost_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            else:
                self.aux_cost_function = nn.CrossEntropyLoss()
                
        else:
            pass

        # scaling temperature for NLL and ECE computation
        self.tuned_scale_T = None

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if self.config.model.arch == "simple" or \
                (self.config.model.arch == "linear" and self.config.data.dataset == "MNIST"):
            x = torch.flatten(x, 1)
        #y_pred = self.cond_pred_model(x)
        y_pred, y_global, y_local = self.cond_pred_model(x)
        return y_pred, y_global, y_local

    def evaluate_guidance_model(self, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set accuracy.
        """
        y_acc_list = []
        for step, feature_label_set in tqdm(enumerate(dataset_loader)):
            # logging.info("\nEvaluating test Minibatch {}...\n".format(step))
            # minibatch_start = time.time()
            x_batch, y_labels_batch = feature_label_set
            y_labels_batch = y_labels_batch.reshape(-1, 1)
            y_pred_prob,_,_ = self.compute_guiding_prediction(
                x_batch.to(self.device))  # (batch_size, n_classes)
            y_pred_prob = y_pred_prob.softmax(dim=1)
            y_pred_label = torch.argmax(y_pred_prob, 1, keepdim=True).cpu().detach().numpy()  # (batch_size, 1)
            y_labels_batch = y_labels_batch.cpu().detach().numpy()
            y_acc = y_pred_label == y_labels_batch  # (batch_size, 1)
            #print(y_acc)
            if len(y_acc_list) == 0:
                y_acc_list = y_acc
            else:
                y_acc_list = np.concatenate([y_acc_list, y_acc], axis=0)
        y_acc_all = np.mean(y_acc_list)
        return y_acc_all

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred,y_global,y_local = self.compute_guiding_prediction(x_batch)
        # y_batch_pred = y_batch_pred.softmax(dim=1)
        #aux_cost = self.aux_cost_function(y_batch_pred, y_batch)+self.aux_cost_function(y_global, y_batch)+self.aux_cost_function(y_local, y_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()
    
    def evaluate_model_on_dataset(self, model, loader, device, config):
        model.eval()
        self.cond_pred_model.eval()
        acc_avg = 0.
        kappa_avg = 0.
        y1_true=None
        y1_pred=None
        for test_batch_idx, (images, target) in enumerate(loader):
            images_unflat = images.to(device)
            if config.data.dataset == "toy" \
                    or config.model.arch == "simple" \
                    or config.model.arch == "linear":
                images = torch.flatten(images, 1)
            images = images.to(device)
            target = target.to(device)
            # target_vec = nn.functional.one_hot(target).float().to(device)
            with torch.no_grad():
                target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                target_pred = target_pred.softmax(dim=1)
                # prior mean at timestep T
                y_T_mean = target_pred
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                    target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                    target_pred = target_pred.softmax(dim=1)

                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                            self.num_timesteps, self.alphas,
                                            self.one_minus_alphas_bar_sqrt,
                                            only_last_sample=True)                               
                y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                y1_true = torch.cat([y1_true, target]) if y1_true is not None else target
                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
        kappa_avg = cohen_kappa(y1_pred.detach().cpu(), y1_true.cpu()).item()
        f1_avg = compute_f1_score(y1_true,y1_pred).item()
                
        acc_avg /= (test_batch_idx + 1)
            
        precision_avg = compute_precision_score(y1_true, y1_pred)
        recall_avg = compute_recall_score(y1_true, y1_pred)
        bacc_avg = compute_bacc_score(y1_true, y1_pred, np.asarray(self.labels_balance))

        return acc_avg, kappa_avg, f1_avg, precision_avg, recall_avg, bacc_avg

    def train(self, fold_n):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(args, config, fold_n)

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        print('successfully load')
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        model = model.to(self.device)
        y_acc_aux_model = self.evaluate_guidance_model(test_loader)
        logging.info("\nBefore training, the guidance classifier accuracy on the test set is {:.8f}.\n\n".format(
            y_acc_aux_model))

        optimizer = get_optimizer(self.config.optim, model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

        self.labels_balance = train_dataset.labels_balance
        labels_balance = np.asarray(train_dataset.labels_balance)
        labels_weight = 1 / labels_balance
        normalized_weight = labels_weight / labels_weight.sum()
        weight_tensor = torch.tensor(normalized_weight, dtype=torch.float).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)

        # apply an auxiliary optimizer for the guidance classifier
        if config.diffusion.apply_aux_cls:
            aux_optimizer = get_optimizer(self.config.aux_optim,
                                          self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):  # load saved auxiliary classifier
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
                self.cond_pred_model.eval()
            elif hasattr(config.diffusion, "trained_aux_cls_log_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_log_path, "aux_ckpt.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
                self.cond_pred_model.eval()
            else:  # pre-train the guidance auxiliary classifier
                assert config.diffusion.aux_cls.pre_train
                self.cond_pred_model.train()
                pretrain_start_time = time.time()
                for epoch in range(config.diffusion.aux_cls.n_pretrain_epochs):
                    for feature_label_set in train_loader:
                        if config.data.dataset == "gaussian_mixture":
                            x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                        else:
                            x_batch, y_labels_batch = feature_label_set
                            y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch,
                                                                                                  config)
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch.to(self.device),
                                                                            y_one_hot_batch.to(self.device),
                                                                            aux_optimizer)
                    if epoch % config.diffusion.aux_cls.logging_interval == 0:
                        logging.info(
                            f"epoch: {epoch}, guidance auxiliary classifier pre-training loss: {aux_loss}"
                        )
                pretrain_end_time = time.time()
                logging.info("\nPre-training of guidance auxiliary classifier took {:.4f} minutes.\n".format(
                    (pretrain_end_time - pretrain_start_time) / 60))
                # save auxiliary model after pre-training
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
            # report accuracy on both training and test set for the pre-trained auxiliary classifier
            y_acc_aux_model = self.evaluate_guidance_model(train_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                y_acc_aux_model))
            y_acc_aux_model = self.evaluate_guidance_model(test_loader)
            logging.info("\nAfter pre-training, guidance classifier accuracy on the test set is {:.8f}.\n".format(
                y_acc_aux_model))

        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                
                if not self.args.reset_epoch:
                    start_epoch = states[2]
                    step = states[3]
                    
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])
                    
            elif self.args.transfer_learning:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.apply_aux_cls and (
                        hasattr(config.diffusion, "trained_aux_cls_ckpt_path") is False) and (
                        hasattr(config.diffusion, "trained_aux_cls_log_path") is False):
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            max_accuracy = 0.0
            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            if args.add_ce_loss:
                logging.info("Apply cross entropy as an auxiliary loss during training.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, feature_label_set in enumerate(train_loader):
                    if config.data.dataset == "gaussian_mixture":
                        x_batch, y_one_hot_batch, y_logits_batch, y_labels_batch = feature_label_set
                    else:
                        x_batch, y_labels_batch = feature_label_set
                        y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch, config)
                    if config.optim.lr_schedule:
                        adjust_learning_rate(optimizer, i / len(train_loader) + epoch, config)
                    n = x_batch.size(0)
                    # record unflattened x as input to guidance aux classifier
                    x_unflat_batch = x_batch.to(self.device)
                    if config.data.dataset == "toy" or config.model.arch in ["simple", "linear"]:
                        x_batch = torch.flatten(x_batch, 1)
                    data_time += time.time() - data_start
                    model.train()
                    self.cond_pred_model.eval()
                    step += 1

                    # antithetic sampling
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    x_batch = x_batch.to(self.device)
                    y_0_hat_batch, y_0_global, y_0_local = self.compute_guiding_prediction(x_unflat_batch)
                    y_0_hat_batch = y_0_hat_batch.softmax(dim=1)
                    y_0_global, y_0_local = y_0_global.softmax(dim=1), y_0_local.softmax(dim=1)

                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_0_hat_batch.shape).to(y_0_hat_batch.device)
                    y_0_batch = y_one_hot_batch.to(self.device)
                    e = torch.randn_like(y_0_batch).to(y_0_batch.device)
                    y_t_batch = q_sample(y_0_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_global = q_sample(y_0_batch, y_0_global,
                                        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    y_t_batch_local = q_sample(y_0_batch, y_0_local,
                                        self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    output = model(x_batch, y_t_batch, t, y_0_hat_batch)
                    output_global = model(x_batch, y_t_batch_global, t, y_0_global)
                    output_local = model(x_batch, y_t_batch_local, t, y_0_local)
                    loss = (e - output).square().mean() + 0.5*(compute_mmd(e,output_global) + compute_mmd(e,output_local))  # use the same noise sample e during training to compute loss
                    # cross-entropy for y_0 reparameterization
                    loss0 = torch.tensor([0])
                    if args.add_ce_loss:
                        y_0_reparam_batch = y_0_reparam(model, x_batch, y_t_batch, y_0_hat_batch, y_T_mean, t,
                                                        self.one_minus_alphas_bar_sqrt)
                        raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                        loss0 = criterion(raw_prob_batch, y_labels_batch.to(self.device))
                        loss += config.training.lambda_ce * loss0

                    if not tb_logger is None:
                        tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (
                                    f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, "
                                    f"Noise Estimation loss: {loss.item()}, " +
                                    f"data time: {data_time / (i + 1)}"
                            )
                        )

                    # optimize diffusion model that predicts eps_theta
                    scheduler.step(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # joint train aux classifier along with diffusion model
                    if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_unflat_batch, y_0_batch,
                                                                            aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, guidance auxiliary classifier joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        # save current states
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                            aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                            if step > 1:  # skip saving the initial ckpt
                                torch.save(
                                    aux_states,
                                    os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                )
                            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    data_start = time.time()

                logging.info(
                    (f"epoch: {epoch}, step: {step}, CE loss: {loss0.item()}, Noise Estimation loss: {loss.item()}, " +
                     f"data time: {data_time / (i + 1)}")
                )


                if epoch % self.config.training.validation_freq == 0 or epoch + 1 == self.config.training.n_epochs:
                    # Evaluate on validation dataset
                    val_acc_avg, val_kappa_avg, val_f1_avg, val_precision_avg, val_recall_avg, val_bacc_avg = self.evaluate_model_on_dataset(
                        model, val_loader, self.device, config)
                    
                    if not tb_logger is None:
                        tb_logger.add_scalar('val_accuracy', val_acc_avg, global_step=step)
                        tb_logger.add_scalar('val_kappa', val_kappa_avg, global_step=step)
                        tb_logger.add_scalar('val_f1', val_f1_avg, global_step=step)
                        tb_logger.add_scalar('val_precision', val_precision_avg, global_step=step)
                        tb_logger.add_scalar('val_recall', val_recall_avg, global_step=step)
                        tb_logger.add_scalar('val_bacc', val_bacc_avg, global_step=step)
                    
                    logging.info(
                        (
                            f"Validation - epoch: {epoch}, step: {step}, "
                            f"Average accuracy: {val_acc_avg}, Average Kappa: {val_kappa_avg}, "
                            f"Average F1: {val_f1_avg}, Precision: {val_precision_avg}, "
                            f"Recall: {val_recall_avg}, Balanced Accuracy: {val_bacc_avg}"
                        )
                    )
                    
                    # Evaluate on test dataset
                    test_acc_avg, test_kappa_avg, test_f1_avg, test_precision_avg, test_recall_avg, test_bacc_avg = self.evaluate_model_on_dataset(
                        model, test_loader, self.device, config)
                    
                    if test_bacc_avg > max_accuracy:
                        logging.info("Update best balanced accuracy at Epoch {}.".format(epoch))
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        torch.save(states, os.path.join(self.args.log_path, "ckpt_best.pth"))
                        aux_states = [
                                self.cond_pred_model.state_dict(),
                                aux_optimizer.state_dict(),
                            ]
                        torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt_best.pth"))
                        
                        best_acc_avg = test_acc_avg
                        best_kappa_avg = test_kappa_avg
                        best_precision_avg = test_precision_avg
                        best_f1_avg = test_f1_avg
                        best_recall_avg = test_recall_avg
                        best_bacc_avg = test_bacc_avg
                    max_accuracy = max(max_accuracy, test_bacc_avg)
                    
                    if not tb_logger is None:
                        tb_logger.add_scalar('test_accuracy', test_acc_avg, global_step=step)
                        tb_logger.add_scalar('test_kappa', test_kappa_avg, global_step=step)
                        tb_logger.add_scalar('test_f1', test_f1_avg, global_step=step)
                        tb_logger.add_scalar('test_precision', test_precision_avg, global_step=step)
                        tb_logger.add_scalar('test_recall', test_recall_avg, global_step=step)
                        tb_logger.add_scalar('test_bacc', test_bacc_avg, global_step=step)
                    
                    logging.info(
                        (
                            f"Test - epoch: {epoch}, step: {step}, "
                            f"Average accuracy: {test_acc_avg}, Average Kappa: {test_kappa_avg}, "
                            f"Average F1: {test_f1_avg}, Precision: {test_precision_avg}, "
                            f"Recall: {test_recall_avg}, Balanced Accuracy: {test_bacc_avg}, "
                            f"Max accuracy: {max_accuracy}"
                        )
                    )

            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.apply_aux_cls and config.diffusion.aux_cls.joint_train:
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(train_loader)
                logging.info("After joint-training, guidance classifier accuracy on the training set is {:.8f}.".format(
                    y_acc_aux_model))
                # report test set accuracy if applied joint training
                y_acc_aux_model = self.evaluate_guidance_model(test_loader)
                logging.info("After joint-training, guidance classifier accuracy on the test set is {:.8f}.".format(
                    y_acc_aux_model))
                
        del data_object
        del train_dataset
        del val_dataset
        del test_dataset
        del train_loader
        del val_loader
        del test_loader
        
        if 'ema_helper' in locals():
            del ema_helper
        if 'aux_optimizer' in locals():
            del aux_optimizer
        
        torch.cuda.empty_cache()
        gc.collect()
                
        return best_acc_avg, best_kappa_avg, best_precision_avg, best_f1_avg, best_recall_avg, best_bacc_avg

    def test(self, fold_n):
        args = self.args
        config = self.config
        data_object, train_dataset, val_dataset, test_dataset = get_dataset(args, config, fold_n)
        log_path = os.path.join(self.args.log_path)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model = ConditionalModel(config, guidance=config.diffusion.include_guidance)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            if args.eval_best:
                ckpt_id = 'best'
                states = torch.load(os.path.join(log_path, f"ckpt_{ckpt_id}.pth"),
                                    map_location=self.device)
            else:
                ckpt_id = 'last'
                states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                    map_location=self.device)
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        #for param in self.cond_pred_model.parameters():
        #    num_params += param.numel()
        #print('Total number of parameters: %d' % num_params)

        # load auxiliary model
        if config.diffusion.apply_aux_cls:
            if hasattr(config.diffusion, "trained_aux_cls_ckpt_path"):
                aux_states = torch.load(os.path.join(config.diffusion.trained_aux_cls_ckpt_path,
                                                     config.diffusion.trained_aux_cls_ckpt_name),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states['state_dict'], strict=True)
            else:
                aux_cls_path = log_path
                if hasattr(config.diffusion, "trained_aux_cls_log_path"):
                    aux_cls_path = config.diffusion.trained_aux_cls_log_path
                aux_states = torch.load(os.path.join(aux_cls_path, "aux_ckpt_best.pth"),
                                        map_location=self.device)
                self.cond_pred_model.load_state_dict(aux_states[0], strict=False)
                logging.info(f"Loading from: {aux_cls_path}/aux_ckpt_best.pth")

        # Evaluate
        model.eval()
        self.cond_pred_model.eval()
        acc_avg = 0.
        kappa_avg = 0.
        y1_true = None
        y1_pred = None
        for test_batch_idx, (images, target) in enumerate(test_loader):
            # if test_batch_idx > 3:
            #     continue
            images_unflat = images.to(self.device)
            images = images.to(self.device)
            target = target.to(self.device)
            target_vec = nn.functional.one_hot(target, num_classes=config.data.num_classes).float().to(self.device)
            with torch.no_grad():
                target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                target_pred = target_pred.softmax(dim=1)
                # prior mean at timestep T
                y_T_mean = target_pred
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean = torch.zeros(target_pred.shape).to(target_pred.device)
                if not config.diffusion.noise_prior:  # apply f_phi(x) instead of 0 as prior mean
                    target_pred, y_global, y_local = self.compute_guiding_prediction(images_unflat)
                    target_pred = target_pred.softmax(dim=1)
                label_t_0 = p_sample_loop(model, images, target_pred, y_T_mean,
                                                self.test_num_timesteps, self.alphas,
                                                self.one_minus_alphas_bar_sqrt,
                                                only_last_sample=True)
                #print(label_t_0.shape)
                label_t_0 = label_t_0.softmax(dim=-1)
                acc_avg += accuracy(label_t_0.detach().cpu(), target.cpu())[0].item()
                kappa_avg += cohen_kappa(label_t_0.detach().cpu(), target.cpu()).item()
                y1_pred = torch.cat([y1_pred, label_t_0]) if y1_pred is not None else label_t_0
                y1_true = torch.cat([y1_true, target]) if y1_true is not None else target                 

        # Getting labels_balance in order to obtain BACC
        labels_balance = np.asarray(train_dataset.labels_balance)

        f1_avg = compute_f1_score(y1_true,y1_pred)
        acc_avg /= (test_batch_idx + 1)
        kappa_avg /= (test_batch_idx + 1)
        precision_avg = compute_precision_score(y1_true, y1_pred)
        recall_avg = compute_recall_score(y1_true, y1_pred)
        bacc_avg = compute_bacc_score(y1_true, y1_pred, labels_balance)
        logging.info(
                        (
                            f"Test - "
                            f"Average accuracy: {acc_avg}, Average Kappa: {kappa_avg}, "
                            f"Average F1: {f1_avg}, Precision: {precision_avg}, "
                            f"Recall: {recall_avg}, Balanced Accuracy: {bacc_avg}"
                        )
                    )
        
        
        return f1_avg, acc_avg, kappa_avg, precision_avg, recall_avg, bacc_avg

