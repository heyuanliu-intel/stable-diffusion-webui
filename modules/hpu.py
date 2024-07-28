import torch

import modules.sd_vae as sd_vae
import modules.images as images
import modules.shared as shared
import modules.sd_models as sd_models

from modules import devices
from modules.shared import opts, state
from modules.sd_models import model_data, model_name_path
from modules.processing import Processed, get_fixed_seed, create_infotext, StableDiffusionProcessing

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

            generator = torch.Generator(device="cpu")
            generator.manual_seed(p.seed + n)

            kwargs = {
                "prompt": p.prompts,
                "negative_prompt": p.negative_prompts,
                "num_inference_steps": p.steps,
                "num_images_per_prompt": 1,
                "height": p.height,
                "width": p.width,
                "guidance_scale": p.cfg_scale if p.cfg_scale else 7,
                "generator": generator,
                "output_type": "pil",
                "throughput_warmup_steps": 0,
                "profiling_warmup_steps": 0,
                "profiling_steps": 0
            }

            if p.eta is not None:
                kwargs["eta"] = p.eta

            result = pipeline(**kwargs)
            for image in result.images:
                images.save_image(image, p.outpath_samples, "")
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
