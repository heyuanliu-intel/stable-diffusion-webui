from typing import Optional
import torch
import torch.nn.functional as F
import diffusers as diffusers
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention

import modules.sd_vae as sd_vae
import modules.images as images
import modules.shared as shared
import modules.sd_models as sd_models

from modules import devices
from modules.shared import opts, state
from modules.sd_models import model_data, model_name_path
from modules.processing import Processed, get_fixed_seed, create_infotext, StableDiffusionProcessing

import habana_frameworks.torch as htorch
from optimum.habana.utils import set_seed
from optimum.habana.diffusers import (GaudiDDIMScheduler, GaudiEulerAncestralDiscreteScheduler, GaudiEulerDiscreteScheduler)


def get_scheduler(scheduler_name):
    # Initialize the scheduler and the generation pipeline
    kwargs = {"timestep_spacing": "linspace"}
    if scheduler_name == "Euler":
        print(f"scheduler_name:{scheduler_name} and using GaudiEulerDiscreteScheduler")
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name_path, subfolder="scheduler", **kwargs)
    elif scheduler_name == "Euler a":
        print(f"scheduler_name:{scheduler_name} and using GaudiEulerAncestralDiscreteScheduler")
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(model_name_path, subfolder="scheduler", **kwargs)
    else:
        print(f"scheduler_name:{scheduler_name} and using GaudiDDIMScheduler")
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name_path, subfolder="scheduler", **kwargs)
    return scheduler


def generate_image_by_hpu(p: StableDiffusionProcessing) -> Processed:
    if isinstance(p.prompt, list):
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    set_seed(seed)

    if p.refiner_checkpoint not in (None, "", "None", "none"):
        p.refiner_checkpoint_info = sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
        if p.refiner_checkpoint_info is None:
            raise Exception(f'Could not find checkpoint with name {p.refiner_checkpoint}')

    if hasattr(shared.sd_model, 'fix_dimensions'):
        p.width, p.height = shared.sd_model.fix_dimensions(p.width, p.height)

    p.sd_model_name = shared.sd_model.sd_checkpoint_info.name_for_extra
    p.sd_model_hash = shared.sd_model.sd_model_hash
    p.sd_vae_name = sd_vae.get_loaded_vae_name()
    p.sd_vae_hash = sd_vae.get_loaded_vae_hash()

    p.fill_fields_from_opts()
    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    pipeline = model_data.sd_model
    pipeline.scheduler = get_scheduler(p.sampler_name)

    infotexts = []
    output_images = []
    with torch.no_grad():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted or state.stopping_generation:
                break

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if len(p.prompts) == 0:
                break

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            kwargs = {
                "prompt": p.prompts,
                "negative_prompt": p.negative_prompts,
                "num_inference_steps": p.steps,
                "num_images_per_prompt": 1,
                "height": p.height,
                "width": p.width,
                "guidance_scale": p.cfg_scale if p.cfg_scale else 7,
                "output_type": "pil",
                "throughput_warmup_steps": 0,
                "profiling_warmup_steps": 0,
                "profiling_steps": 0
            }

            if p.eta is not None:
                kwargs["eta"] = p.eta

            result = pipeline(**kwargs)
            output_images += result.images

            result.images = None
            result = None
            devices.torch_gc()
            state.nextjob()

    if not infotexts:
        infotexts.append(Processed(p, []).infotext(p, 0))

    def infotext(index=0, use_main_prompt=False):
        return create_infotext(p, p.prompts, p.seeds, p.subseeds, use_main_prompt=use_main_prompt, index=index, all_negative_prompts=p.negative_prompts)

    index_of_first_image = 0
    unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
    if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
        grid = images.image_grid(output_images, p.batch_size)

        if opts.return_grid:
            text = infotext(use_main_prompt=True)
            infotexts.insert(0, text)
            if opts.enable_pnginfo:
                grid.info["parameters"] = text
            output_images.insert(0, grid)
            index_of_first_image = 1
        if opts.grid_save:
            images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(use_main_prompt=True), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    devices.torch_gc()

    return Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        comments="",
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # print("#####################################scaled_dot")
        # hidden_states = F.scaled_dot_product_attention(
        #    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        import habana_frameworks.torch.hpu as ht
        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        with ht.sdp_kernel(enable_recompute=False):
            hidden_states = FusedSDPA.apply(query, key, value, None, 0.0, False, None)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


diffusers.models.attention_processor.AttnProcessor2_0 = AttnProcessor2_0
