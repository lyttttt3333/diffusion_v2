from typing import Dict, OrderedDict

# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

# from robomimic.algo import algo_factory
# from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as rmoc
from robomimic.models.obs_nets import ObservationGroupEncoder
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        feat_proc="none",  # 'none', 'linear', 'mlp'
        compositional=True,
        part_decomp=False,
        output_type="pose",
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        assert not (part_decomp and compositional)  # cannot be both

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {
            "low_dim": [],
            "rgb": [],
            "depth": [],
            "scan": [],
            "spatial": [],
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            elif type == "spatial":
                obs_config["spatial"].append(key)
            elif type == "depth":
                obs_config["depth"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(
            config.observation.encoder
        )
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(obs_key_shapes)

        # update arg for feature processing
        if feat_proc == "none":
            encoder_kwargs["spatial"]["core_kwargs"]["use_feats"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["proj_feats"] = False
        elif feat_proc == "linear":
            encoder_kwargs["spatial"]["core_kwargs"]["use_feats"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["proj_feats"] = True
        elif feat_proc == "mlp":
            encoder_kwargs["spatial"]["core_kwargs"]["use_feats"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["preproc_feats"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["proj_feats"] = False
        elif feat_proc == "distill":
            encoder_kwargs["spatial"]["core_kwargs"]["use_feats"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["preproc_feats"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["proj_feats"] = False
        else:
            raise ValueError(f"Unsupported feat_proc: {feat_proc}")

        # update arg for compositional
        # when using seg, must be compositional
        # if "d3fields" in shape_meta["obs"]:
        #     use_seg = shape_meta["obs"]["d3fields"]["info"]["use_seg"]
        #     if use_seg:
        #         assert compositional
        #     else:
        #         assert not compositional

        if compositional:
            encoder_kwargs["spatial"]["core_kwargs"]["compositional"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["use_pos"] = True
            encoder_kwargs["spatial"]["core_kwargs"]["decomp_part"] = False
        elif part_decomp:
            encoder_kwargs["spatial"]["core_kwargs"]["compositional"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["use_pos"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["decomp_part"] = True
        else:
            encoder_kwargs["spatial"]["core_kwargs"]["compositional"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["use_pos"] = False
            encoder_kwargs["spatial"]["core_kwargs"]["decomp_part"] = False

        # update arg for n_per_obj
        if "d3fields" in shape_meta["obs"]:
            encoder_kwargs["spatial"]["core_kwargs"]["n_per_obj"] = shape_meta["obs"][
                "d3fields"
            ]["info"]["N_per_inst"]

        encoder = ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        obs_encoder = encoder.nets["obs"]

        ### DEPRECATED: use robomimic algo instead
        # policy: PolicyAlgo = algo_factory(
        #         algo_name=config.algo_name,
        #         config=config,
        #         obs_key_shapes=obs_key_shapes,
        #         ac_dim=action_dim,
        #         device='cpu',
        #     )

        # obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            # output_type=output_type,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.output_type = output_type
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print(
            "Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters())
        )

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        diffusion_proc_list_multi_seed = list()
        # torch.random.manual_seed(0)

        for iter in range(1):

            diffusion_proc_list = list()

            trajectory = torch.randn(
                size=condition_data.shape,
                dtype=condition_data.dtype,
                device=condition_data.device,
                generator=generator,
            )

            # set step values
            scheduler.set_timesteps(self.num_inference_steps)

            for t in scheduler.timesteps:
                diffusion_proc_list.append(trajectory)
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = model(
                    trajectory, t, local_cond=local_cond, global_cond=global_cond
                )

                # 3. compute previous image: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output, t, trajectory, generator=generator, **kwargs
                ).prev_sample

            # finally make sure conditioning is enforced
            trajectory[condition_mask] = condition_data[condition_mask]

            diffusion_proc_list.append(trajectory)
            diffusion_proc_list = torch.cat(diffusion_proc_list, dim=0)

            diffusion_proc_list_multi_seed.append(diffusion_proc_list[None,...])
        
        diffusion_proc_list_multi_seed = torch.cat(diffusion_proc_list_multi_seed, dim=0)

        return trajectory, diffusion_proc_list_multi_seed

    def extract_pcd_heatmap(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        cam_pn = self.obs_encoder.obs_nets["d3fields"].nets
        pn_feats, pn_heatmap = cam_pn(this_nobs["d3fields"])

        return pn_heatmap

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample, nsample_proc = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # unnormalize prediction proc
        naction_pred_proc = nsample_proc[..., :Da]
        action_pred_proc = self.normalizer["action"].unnormalize(naction_pred_proc)

        # get action
        # if self.output_type == "pose" or self.output_type == "progress":
        #     start = To - 1
        # else:
        #     start = 0
        start = To - 1
        end = min(start + self.n_action_steps, action_pred.shape[1])
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred, "action_pred_proc": action_pred_proc}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(
            trajectory.shape
        )  # NOTE: what is mask_generator - seems like something abandoned

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
