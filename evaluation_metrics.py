import multiprocessing
import os
from argparse import ArgumentParser

import numpy as np
import torch
from absl import logging
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import functional as F

from dataset_tools.dataset_templates import create_dataset
from utils import set_logger


def evaluate_quantitative_scores_text2img(
    pipe,
    real_image_path,
    mscoco_anno,
    n_images=5000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",  # reuse_generated=True,
    negative_prompt="",
    guidance_scale=4.5,
    name_format="pad_png",
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    if real_image_path is not None:
        fid_value = calculate_fid_given_paths(
            [real_image_path, fake_image_path],
            1,  # 64,
            device,
            dims=2048,
            num_workers=0,  # 8,
        )
        results["FID"] = fid_value
        print(f"FID: {fid_value}")

    # Inception Score
    inception = InceptionScore().to(device)
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    # if os.path.exists(fake_image_path) and not reuse_generated:
    #     os.system(f"rm -rf {fake_image_path}")
    # os.makedirs(fake_image_path, exist_ok=True)

    img_type = name_format.split("_")[1]

    for index in range(0, n_images, batchsize):

        slice = mscoco_anno["annotations"][index : index + batchsize]
        print(f"Processing {index}th image")
        caption_list = [d["caption"] for d in slice]

        filename_list = []
        for d in slice:
            img_name = str(d["id"])
            if name_format.split("_")[0] == "pad":
                img_name = img_name.zfill(12)

            filename_list.append(img_name)

        torch_images = []
        for filename in filename_list:
            image_file = f"{fake_image_path}/{filename}.{img_type}"
            if os.path.exists(image_file):
                image = Image.open(image_file)
                image_np = np.array(image)
                torch_image = torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2)
                torch_images.append(torch_image)
            else:
                print(image_file)

        if len(torch_images) > 0:
            torch_images = torch.cat(torch_images, dim=0)
            print(torch_images.shape)
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list[: len(torch_images)])
        else:
            output = pipe(
                caption_list,
                generator=generator,
                output_type="np",
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
            )
            fake_images = output.images
            # Inception Score
            count = 0
            torch_images = (
                torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
            )
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list)
            for j, image in enumerate(fake_images):
                # image = image.astype(np.uint8)
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{fake_image_path}/{filename_list[count]}.jpg")
                count += 1

    IS = inception.compute()
    CLIP = clip.compute()
    results["IS"] = IS
    results["CLIP"] = CLIP
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")

    return results


if __name__ == "__main__":

    # set start method as 'spawn' to avoid CUDA re-initialization issues
    multiprocessing.set_start_method("spawn")

    parser = ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=str,
        default="./workdir_parti-16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="parti_cocoformat",  # coco
    )
    parser.add_argument(
        "--dataset_anno_file",
        type=str,
        default="./data/PartiPrompts.tsv",  # 'data/coco/annotations/captions_val2017.json'
    )

    args = parser.parse_args()
    workdir = args.workdir
    annFile = args.dataset_anno_file
    dataset_name = args.dataset_name

    gpu_id = 0
    gpu_ids = [
        0,
    ]
    node_id = 0
    node_ids = [
        0,
    ]
    dataset_params = dict(
        name=dataset_name,
        annFile=annFile,
        ds_type="eval",
    )

    name_format = "nopad_png"

    set_logger(log_level="info", fname=os.path.join(workdir, "output.log"))

    ds = create_dataset(
        gpu_id=gpu_id,
        gpu_ids=gpu_ids,
        node_id=node_id,
        node_ids=node_ids,
        **dataset_params,
    )

    real_image_path = ds.root if hasattr(ds, "root") else None

    n_images = len(ds.anno["annotations"])

    results = evaluate_quantitative_scores_text2img(
        pipe=None,
        real_image_path=real_image_path,
        mscoco_anno=ds.anno,
        n_images=n_images,
        batchsize=1,
        seed=1,
        fake_image_path=workdir,
        name_format=name_format,
    )
    for k, v in results.items():
        logging.info(f"{k}: {v}")
