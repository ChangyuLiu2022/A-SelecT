#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os

from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self.weight_dtype = self.model.weight_dtype

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        if self.cfg.MODEL.PROMPT_HEAD_PATH:
            self.load_prompt_and_head()
            logger.info(f"Prompt and head loaded from {cfg.MODEL.PROMPT_HEAD_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, timesteps_list, prompt_embeds, pooled_prompt_embeds, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        #inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs[0].shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs, timesteps_list, prompt_embeds, pooled_prompt_embeds)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        #print(f"t7: {time.time()}")

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        #self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        # self.cls_weights = train_loader.dataset.get_class_weights(
        #     self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        self.cls_weights = None
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training




        #get unconditional propmt embeddings
        #some inputs for pipeline
        prompt=[""] * self.cfg.DATA.BATCH_SIZE
        prompt_2=None
        prompt_3=None
        negative_prompt=""
        negative_prompt_2=None
        negative_prompt_3=None
        num_inference_steps=1000
        height=self.cfg.DATA.CROPSIZE
        width=self.cfg.DATA.CROPSIZE
        max_sequence_length=256
        timesteps=None
        prompt_embeds=None
        negative_prompt_embeds=None
        pooled_prompt_embeds=None
        negative_pooled_prompt_embeds=None


        #Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.model.pipe.scheduler, num_inference_steps, self.device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.model.pipe.scheduler.order, 0)
        self.model.pipe._num_timesteps = len(timesteps)
        self.model.pipe.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            callback_on_step_end_tensor_inputs=["latents"],
            max_sequence_length=max_sequence_length,
        )
        self.model.pipe._clip_skip = None
        self.model.pipe._joint_attention_kwargs = None
        self.model.pipe._interrupt = False
        
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=False,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=self.device,
            clip_skip=self.model.pipe.clip_skip,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        self.prompt_embeds = prompt_embeds.detach()
        self.pooled_prompt_embeds = pooled_prompt_embeds.detach()
        if negative_prompt_embeds is not None:
            self.negative_prompt_embeds = negative_prompt_embeds.detach()
        if negative_pooled_prompt_embeds is not None:
            self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.detach()

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                # measure data loading time
                data_time.update(time.time() - end)

                #apply vae encoder and add noise
                X = X.to(self.device)
                X = X.to(dtype=self.weight_dtype)
                #print(f"t0: {time.time()}")
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:

                    model_input = self.model.pipe.vae.encode(X).latent_dist.sample()
                    model_input = model_input * self.model.pipe.vae.config.scaling_factor
                    model_input = model_input.to(dtype=self.weight_dtype)

                    #add noise
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)


                    
                    noisy_model_input_list = []
                    timesteps_list = []
                    for indices in self.cfg.MODEL.T_LIST:
                        # Get the timestep and sigma for each sample in the batch
                        timestep = self.model.pipe.scheduler.timesteps[indices].to(device=model_input.device)
                        timesteps = timestep.expand(model_input.shape[0])
                        # Add noise according to flow matching.
                        sigmas = self.model.pipe.scheduler.sigmas[indices].to(device=model_input.device)
                        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
                        noisy_model_input_list.append(noisy_model_input)
                        timesteps_list.append(timesteps)

                train_loss, _ = self.forward_one_batch(noisy_model_input_list, targets, timesteps_list, 
                                                        self.prompt_embeds[:model_input.shape[0]], self.pooled_prompt_embeds[:model_input.shape[0]], True)

                if train_loss == -1:
                    #continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()
            #self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            # if test_loader is not None:
            #     self.eval_classifier(
            #         test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            #t_name = "val_" + val_loader.dataset.name
            t_name = "val_" + self.cfg.DATA.NAME
            test_name = "test_" + self.cfg.DATA.NAME
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
                if test_loader is not None and curr_acc > self.cfg.SOLVER.ACC_THRESHOLD:
                    #make sure acc is good enough to run test
                    self.eval_classifier(
                        test_loader, "test", epoch == total_epoch - 1)

                #save the best prompt and save the best head
                if self.cfg.MODEL.SAVE_CKPT:
                    self.save_prompt_and_head(best_epoch)
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break
        
        #save the best results info
        logger.info(
                    f'Best epoch {best_epoch}: val top1: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][t_name]["top1"]:.3f} top5: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][t_name]["top5"]:.3f}')
        logger.info(
                    f'test top1: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][test_name]["top1"]:.3f} top5: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][test_name]["top5"]:.3f}')
        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")



    @torch.no_grad()
    def save_prompt_and_head(self, epoch):
        out = {"shallow_prompt": self.model.prompt_embeddings}
        if self.cfg.MODEL.PROMPT.DEEP:
            out["deep_prompt"] = self.model.deep_prompt_embeddings
        out["head"] = self.model.head.state_dict()
        out["prompt_proj"] = self.model.prompt_proj.state_dict()
        out["epoch"] = epoch    
        torch.save(out, os.path.join(
            self.cfg.OUTPUT_DIR, f"best_prompt_head_ep{epoch}.pth"))
    
    def load_prompt_and_head(self):
        out = torch.load(self.cfg.MODEL.PROMPT_HEAD_PATH)
        self.model.prompt_embeddings = out["shallow_prompt"].to(self.device)
        if self.cfg.MODEL.PROMPT.DEEP:
            self.model.deep_prompt_embeddings = out["deep_prompt"].to(self.device)
        self.model.head.load_state_dict(out["head"])
        self.model.prompt_proj.load_state_dict(out["prompt_proj"])
        

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        #test_name = prefix + "_" + data_loader.dataset.name
        test_name = prefix + "_" + self.cfg.DATA.NAME
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
           
            # measure data loading time
            data_time.update(time.time() - end)
            X, targets = self.get_input(input_data)
            #X, targets = input_data[0], input_data[1]
            X = X.to(self.device)
            X = X.to(dtype=self.weight_dtype)
            with torch.no_grad():
                    # Map input images to latent space + normalize latents:

                    model_input = self.model.pipe.vae.encode(X).latent_dist.sample()
                    model_input = model_input * self.model.pipe.vae.config.scaling_factor
                    model_input = model_input.to(dtype=self.weight_dtype)

                    #add noise
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)


                    
                    noisy_model_input_list = []
                    timesteps_list = []
                    for indices in self.cfg.MODEL.T_LIST:
                        # Get the timestep and sigma for each sample in the batch
                        timestep = self.model.pipe.scheduler.timesteps[indices].to(device=model_input.device)
                        timesteps = timestep.expand(model_input.shape[0])
                        # Add noise according to flow matching.
                        sigmas = self.model.pipe.scheduler.sigmas[indices].to(device=model_input.device)
                        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
                        noisy_model_input_list.append(noisy_model_input)
                        timesteps_list.append(timesteps)
            loss, outputs = self.forward_one_batch(noisy_model_input_list, targets, timesteps_list, 
                                                        self.prompt_embeds[:model_input.shape[0]], self.pooled_prompt_embeds[:model_input.shape[0]], False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        # if self.model.side is not None:
        #     logger.info(
        #         "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
             joint_logits, total_targets,
             test_name, self.cfg.DATA.MULTILABEL,
         )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
    
    def add_noise(self, x, t): 
        #add a noise to x
        noise = torch.randn_like(x).to(self.device)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        return x_t
    


class Trainer_old():
    """
    For original DiT
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        diffusion,
        vae: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        if self.cfg.MODEL.PROMPT_HEAD_PATH:
            self.load_prompt_and_head()
            logger.info(f"Prompt and head loaded from {cfg.MODEL.PROMPT_HEAD_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")

    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        #inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs[0].shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            outputs = self.model(inputs)  # (batchsize, num_cls)
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        outputs.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        #self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        # self.cls_weights = train_loader.dataset.get_class_weights(
        #     self.cfg.DATA.CLASS_WEIGHTS_TYPE)

        self.cls_weights = None
        # logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()

            end = time.time()

            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations
                    break
                
                X, targets = self.get_input(input_data)
                #X, targets = input_data[0], input_data[1]
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                #apply vae encoder and add noise
                X = X.to(self.device)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    X = self.vae.encode(X).latent_dist.sample().mul_(0.18215)
                
                X_t_list = []
                for t in self.cfg.MODEL.T_LIST:
                    t = torch.tensor([t] * X.size(0)).to(self.device) 
                    X_t = self.add_noise(X, t)
                    X_t_list.append(X_t)
                #train_loss, _ = self.forward_one_batch(X_1, targets, True)
                train_loss, _ = self.forward_one_batch(X_t_list, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()
            #self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            # if test_loader is not None:
            #     self.eval_classifier(
            #         test_loader, "test", epoch == total_epoch - 1)

            # check the patience
            #t_name = "val_" + val_loader.dataset.name
            t_name = "val_" + self.cfg.DATA.NAME
            test_name = "test_" + self.cfg.DATA.NAME
            try:
                curr_acc = self.evaluator.results[f"epoch_{epoch}"]["classification"][t_name]["top1"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
                if test_loader is not None:
                    self.eval_classifier(
                        test_loader, "test", epoch == total_epoch - 1)

                #save the best prompt and save the best head
                if self.cfg.MODEL.SAVE_CKPT:
                    self.save_prompt_and_head(best_epoch)
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break
        
        #save the best results info
        logger.info(
                    f'Best epoch {best_epoch}: val top1: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][t_name]["top1"]:.3f} top5: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][t_name]["top5"]:.3f}')
        logger.info(
                    f'test top1: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][test_name]["top1"]:.3f} top5: {self.evaluator.results[f"epoch_{best_epoch-1}"]["classification"][test_name]["top5"]:.3f}')
        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    # @torch.no_grad()
    # def save_prompt(self, epoch):
    #     # only save the prompt embed if below conditions are satisfied
    #     if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
    #         if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
    #             prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
    #             out = {"shallow_prompt": prompt_embds}
    #             if self.cfg.MODEL.PROMPT.DEEP:
    #                 deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
    #                 out["deep_prompt"] = deep_embds
    #             torch.save(out, os.path.join(
    #                 self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

    @torch.no_grad()
    def save_prompt_and_head(self, epoch):
        out = {"shallow_prompt": self.model.prompt_embeddings}
        if self.cfg.MODEL.PROMPT.DEEP:
            out["deep_prompt"] = self.model.deep_prompt_embeddings
        out["head"] = self.model.head.state_dict()
        out["prompt_proj"] = self.model.prompt_proj.state_dict()
        out["epoch"] = epoch    
        torch.save(out, os.path.join(
            self.cfg.OUTPUT_DIR, f"best_prompt_head_ep{epoch}.pth"))
    
    def load_prompt_and_head(self):
        out = torch.load(self.cfg.MODEL.PROMPT_HEAD_PATH)
        self.model.prompt_embeddings = out["shallow_prompt"].to(self.device)
        if self.cfg.MODEL.PROMPT.DEEP:
            self.model.deep_prompt_embeddings = out["deep_prompt"].to(self.device)
        self.model.head.load_state_dict(out["head"])
        self.model.prompt_proj.load_state_dict(out["prompt_proj"])
        

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        #test_name = prefix + "_" + data_loader.dataset.name
        test_name = prefix + "_" + self.cfg.DATA.NAME
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []

        for idx, input_data in enumerate(data_loader):
            end = time.time()
           
            # measure data loading time
            data_time.update(time.time() - end)
            X, targets = self.get_input(input_data)
            #X, targets = input_data[0], input_data[1]
            X = X.to(self.device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                X = self.vae.encode(X).latent_dist.sample().mul_(0.18215)
            
            X_t_list = []
            for t in self.cfg.MODEL.T_LIST:
                t = torch.tensor([t] * X.size(0)).to(self.device)    #t=0
                X_t = self.add_noise(X, t)
                X_t_list.append(X_t)
            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs = self.forward_one_batch(X_t_list, targets, False)
            if loss == -1:
                return
            losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)
        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        # if self.model.side is not None:
        #     logger.info(
        #         "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        self.evaluator.classify(
             joint_logits, total_targets,
             test_name, self.cfg.DATA.MULTILABEL,
         )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
    
    def add_noise(self, x, t): 
        #add a noise to x
        noise = torch.randn_like(x).to(self.device)
        x_t = self.diffusion.q_sample(x, t, noise=noise)
        return x_t