import sys

sys.path.append("./lumina_mgpt/")
sys.path.append("./")

from lumina_mgpt.inference_solver import FlexARInferenceSolver
from scheduler.sjd_pac_iteration_lumina_mgpt import renew_pipeline_sampler


def load_lumina_mgpt(
    cache_dir="./ckpts",
    model_name="Alpha-VLLM/Lumina-mGPT-7B-768",
    target_size=768,
    seed=1,
    max_num_new_tokens=64,
    tree_width=3,
    tree_depth=3,
    guidance_scale=3.0,
    device="cpu",
    **kwargs,
):
    model_path = model_name

    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        precision="bf16",
        target_size=target_size,
        cache_dir=cache_dir,
        device=device,
    )

    print(inference_solver.__class__)
    inference_solver = renew_pipeline_sampler(
        inference_solver,
        jacobi_loop_interval_l=1,
        jacobi_loop_interval_r=(target_size // 16) ** 2 + target_size // 16 - 10,
        max_num_new_tokens=max_num_new_tokens,
        tree_width=tree_width,
        tree_depth=tree_depth,
        guidance_scale=guidance_scale,
        seed=seed,
        do_cfg=True,
        **kwargs,
    )

    return inference_solver


def load_pretrained_model(
    model_name="Alpha-VLLM/Lumina-mGPT-7B-768",
    **kwargs,
):
    if "lumina-mgpt" in model_name.lower():
        return load_lumina_mgpt(model_name=model_name, **kwargs)
    else:
        raise NotImplementedError(
            f"SJD-PAC currently only supports Lumina-mGPT, got: {model_name}"
        )


def get_lumina_mgpt_forward_func(
    inference_solver,
    guidance_scale=3.0,
    image_top_k=2000,
    max_gen_len=8192,
    temperature=1.0,
    target_size=768,
    **kwargs,
):

    def sample_fn(prompts):
        prompts = (
            f"Generate an image of {target_size}x{target_size} according to the following prompt:\n"
            + prompts
        )

        generated = inference_solver.generate(
            images=[],
            qas=[[prompts, None]],
            max_gen_len=max_gen_len,
            temperature=temperature,
            logits_processor=inference_solver.create_logits_processor(
                cfg=guidance_scale, image_top_k=image_top_k
            ),
        )
        a1, new_image = generated[0], generated[1][0]

        result_image = inference_solver.create_image_grid([new_image], 1, 1)
        return result_image

    return sample_fn


def get_forward_func(model_name, model, **kwargs):
    if "lumina-mgpt" in model_name.lower():
        return get_lumina_mgpt_forward_func(model, **kwargs)
    else:
        raise NotImplementedError(
            f"SJD-PAC currently only supports Lumina-mGPT, got: {model_name}"
        )
