import torch
import modules.images as images
from modules.shared import opts
from modules.sd_models import model_data, model_name_or_path
from modules.processing import Processed, get_fixed_seed, create_infotext, StableDiffusionProcessing

from optimum.habana.utils import set_seed
from optimum.habana.diffusers import (GaudiDDIMScheduler, GaudiEulerAncestralDiscreteScheduler, GaudiEulerDiscreteScheduler)


def get_scheduler(scheduler_name):
    # Initialize the scheduler and the generation pipeline
    kwargs = {"timestep_spacing": "linspace"}
    if scheduler_name == "Euler":
        print("scheduler_name:{scheduler_name} and using GaudiEulerDiscreteScheduler")
        scheduler = GaudiEulerDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler", **kwargs)
    elif scheduler_name == "Euler a":
        print("scheduler_name:{scheduler_name} and using GaudiEulerAncestralDiscreteScheduler")
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler", **kwargs)
    else:
        print("scheduler_name:{scheduler_name} and using GaudiDDIMScheduler")
        scheduler = GaudiDDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler", **kwargs)
    return scheduler


def generate_image_by_hpu(p: StableDiffusionProcessing):
    if isinstance(p.prompt, list):
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    p.setup_prompts()

    if isinstance(seed, list):
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    set_seed(p.all_seeds[0])

    if isinstance(subseed, list):
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    pipeline = model_data.sd_model
    pipeline.scheduler = get_scheduler(p.sampler_name)

    kwargs_call = {}
    kwargs_common = {
        "width": p.width,
        "height": p.height,
        "num_images_per_prompt": p.batch_size if p.batch_size else 1,
        "batch_size": 1,
        "num_inference_steps": p.steps if p.steps else 20,
        "guidance_scale": p.cfg_scale if p.cfg_scale else 7,
        "eta": p.eta if p.eta else 0,
        "negative_prompt": p.negative_prompts,
        "throughput_warmup_steps": 0,
        "profiling_warmup_steps": 0,
        "profiling_steps": 0
    }

    print(f"kwargs_common:{kwargs_common}")
    generator = torch.Generator(device="cpu").manual_seed(0)
    kwargs_call["generator"] = generator
    kwargs_call.update(kwargs_common)
    outputs = pipeline(prompt=p.prompt, **kwargs_call)

    infotexts = []
    output_images = outputs.images
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

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0],
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    return res