from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length
import torch.nn as nn


class rDPOTrainer(DPOTrainer):
    def __init__(self, *args, sft_weight=0.0, gamma_beta_ratio=0.0, alpha=1.0, loss='simpo', **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_weight = sft_weight
        self.gamma_beta_ratio = gamma_beta_ratio
        self.alpha = alpha
        self.loss = loss
        
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        # concatenated_batch["concatenated_images"] = batch["images"] + batch["images"]

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        chosen_batch = concatenated_batch["concatenated_input_ids"][:len_chosen]
        rejected_batch = concatenated_batch["concatenated_input_ids"][len_chosen:]
        chosen_mask = concatenated_batch["concatenated_attention_mask"][:len_chosen]
        rejected_mask = concatenated_batch["concatenated_attention_mask"][len_chosen:]
        chosen_label = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_label = concatenated_batch["concatenated_labels"][len_chosen:]

        chosen_model_kwargs = (
            {
                "labels": chosen_label,
                "decoder_input_ids": concatenated_batch.pop("chosen_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )    
        rejected_model_kwargs = (
            {
                "labels": rejected_label,
                "decoder_input_ids": concatenated_batch.pop("rejected_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        # model_kwargs = {
        #     "images": concatenated_batch["concatenated_images"],
        #     "labels": concatenated_batch["concatenated_labels"],
        # }

        # outputs, refined_labels = model(
        #     concatenated_batch["concatenated_input_ids"],
        #     attention_mask=concatenated_batch["concatenated_attention_mask"],
        #     **model_kwargs,
        # )
        # all_logits = outputs.logits.to(torch.float32)

        # all_logps = self._get_batch_logps(
        #     all_logits,
        #     refined_labels,
        #     average_log_prob=False,
        # )

        # chosen_logps = all_logps[:len_chosen]
        # rejected_logps = all_logps[len_chosen:]

        # chosen_logits = all_logits[:len_chosen]
        # rejected_logits = all_logits[len_chosen:]

        # imageless_model_kwargs = {
        #         "labels": batch["chosen_labels"],
        #         "images": batch["image"],
        #         "mask_visual_tokens": True,
        #     }
            
        # imageless_chosen_outputs, imageless_chosen_label = model(
        #     batch["chosen_input_ids"],
        #     attention_mask=batch["chosen_attention_mask"],
        #     **imageless_model_kwargs,
        # )
        if self.loss == 'simpo':
            is_average = True
        elif self.loss == 'dpo':
            is_average = False
        else:
            raise NotImplementedError(f'Unknown loss type: {self.loss}')
        
        chosen_logits = model(
            input_ids = chosen_batch,
            labels = chosen_label,
            images=batch['images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = chosen_batch,
                position_ids = None,
                attention_mask = chosen_mask,
                past_key_values = None,
                labels = chosen_label,
                images = batch['images']
            )

        chosen_logps = self._get_batch_logps(
            chosen_logits,
            new_chosen_labels,
            average_log_prob=is_average,
        )

        rejected_logits = model(
            input_ids = rejected_batch,
            labels = rejected_label,
            images=batch['images'],
            attention_mask=rejected_mask,
            **rejected_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_rejected_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = rejected_batch,
                position_ids = None,
                attention_mask = rejected_mask,
                past_key_values = None,
                labels = rejected_label,
                images = batch['images']
            )

        rejected_logps = self._get_noisy_batch_logps(
            rejected_logits,
            rejected_logits,
            new_rejected_labels,
            average_log_prob=is_average,
        )


        # imageless_model_kwargs = {
        #         "labels": batch["chosen_labels"],
        #         "images": batch["retrieved_images"],
        #     }
        
        # imageless_chosen_outputs, imageless_chosen_label = model(
        #     batch["chosen_input_ids"],
        #     attention_mask=batch["chosen_attention_mask"],
        #     **imageless_model_kwargs,
        # )

        # imageless_chosen_logits = imageless_chosen_outputs.logits.to(torch.float32)

        # imageless_chosen_logps = self._get_batch_logps(
        #     imageless_chosen_logits,
        #     imageless_chosen_label,
        #     average_log_prob=False,
        # )

        imageless_chosen_logits = model(
            input_ids = chosen_batch,
            labels = chosen_label,
            images=batch['retrieved_images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_imageless_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = chosen_batch,
                position_ids = None,
                attention_mask = chosen_mask,
                past_key_values = None,
                labels = chosen_label,
                images = batch['retrieved_images']
            )

        imageless_chosen_logps = self._get_noisy_batch_logps(
            imageless_chosen_logits,
            imageless_chosen_logits,
            new_imageless_chosen_labels,
            average_log_prob=is_average,
        )

        return (chosen_logps, rejected_logps, imageless_chosen_logps, chosen_logits, rejected_logits, imageless_chosen_logits)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless_chosen_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_imageless_chosen_logps: torch.FloatTensor, 
        reference_free: bool = False,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios  # response preference

        image_conditional_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
        image_conditional_ref_logratios = reference_chosen_logps - reference_imageless_chosen_logps

        if reference_free:
            image_conditional_ref_logratios = 0

        image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

        # anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference

        # mDPO 
        losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) 
            # \
            # -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 

        # losses -= policy_chosen_logps / 1024
        
        # KL penalty
        kl =  torch.exp(reference_chosen_logps) * (reference_chosen_logps - policy_chosen_logps)
        # losses += 0.05*kl 

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        imageless_rewards = (
            self.beta * (policy_imageless_chosen_logps - reference_imageless_chosen_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, imageless_rewards, kl

    '''rSimPO = SimPO + alpha * vSimPO '''
    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless_chosen_logps: torch.FloatTensor,
    ):  
        ## TODO these logps should be averaged, set bool True in forward 
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        logits = pi_logratios - self.gamma_beta_ratio ## TODO set gamma_beta_ratio, check what is it
        imageless_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
        image_conditional_logits = imageless_pi_logratios - self.gamma_beta_ratio ## set imageless_gamma_beta_ratio
        
        losses =  -torch.nn.functional.logsigmoid(self.beta * logits) \
            -self.alpha * torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) ## TODO set alpha as a hyperparameter
            
        chosen_rewards = self.beta * policy_chosen_logps.detach()
        rejected_rewards = self.beta * policy_rejected_logps.detach()
        imageless_rewards = self.beta * policy_imageless_chosen_logps.detach()

        kl = torch.zeros_like(chosen_rewards) 

        return losses, chosen_rewards, rejected_rewards, imageless_rewards, kl

        
        
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_imageless_chosen_logits,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_imageless_chosen_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_imageless_chosen_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)
                
        if self.loss == 'dpo':
            losses, chosen_rewards, rejected_rewards, imageless_rewards, kl = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_imageless_chosen_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_imageless_chosen_logps,
            )
        elif self.loss == 'simpo':
            losses, chosen_rewards, rejected_rewards, imageless_rewards, kl = self.simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                policy_imageless_chosen_logps,
            )
        else:
            raise NotImplementedError(f"Unknown loss type:{self.loss}")
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float() ## for simpo it is consistency
        imageless_reward_accuracies = (chosen_rewards > imageless_rewards).float()

        loss = losses.mean()

        chosen_labels = batch["chosen_labels"]

        prefix = "eval_" if train_eval == "eval" else ""

        if self.sft_weight > 0.0: #TODO sft_weight can decay over time
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()

        
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless_chosen"] = imageless_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = imageless_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (chosen_rewards - imageless_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless_chosen"] = policy_imageless_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless_chosen"] = policy_imageless_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}kl div"] = kl.cpu().mean()

        return loss, metrics