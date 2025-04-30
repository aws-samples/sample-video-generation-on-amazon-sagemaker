import os
import boto3
from diffusers import AutoencoderKLCogVideoX, CogVideoXPipeline, CogVideoXTransformer3DModel, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import torch
from transformers import T5EncoderModel

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
region = os.environ["region"]
cloudwatch = boto3.client('cloudwatch', region_name=region)
num_inference_steps = 50

text_path = os.environ.get('text_path', '/opt/ml/processing/input/text')
image_path = os.environ.get('image_path', '/opt/ml/processing/input/image')
output_dir = "/opt/ml/processing/output/generated_video.mp4"

prompt_filename = os.environ["prompt_filename"]
prompt_file_path = os.path.join(text_path, prompt_filename)

if "image_filename" in os.environ.keys():
    image_filename = os.environ["image_filename"]
else:
    image_filename = None
logger.info(f"image filename: {image_filename}")


def callback_fn(pipe, step, timestep, callback_kwargs):
    progress = (40 + (step / num_inference_steps) * 60)/100

    cloudwatch.put_metric_data(
        Namespace='/aws/sagemaker/ProcessingJobs',
        MetricData=[
            {
                'MetricName': 'ProgressPercentage',
                'Value': progress,
                'Unit': 'Percent',
                'Dimensions': [
                    {
                        'Name': 'ProcessingJobName',
                        'Value': os.environ['job_name']
                    }
                ]
            }
        ]
    )
    return {}

def emit_progress_metric(percentage):
    cloudwatch.put_metric_data(
        Namespace='/aws/sagemaker/ProcessingJobs',
        MetricData=[
            {
                'MetricName': 'ProgressPercentage',
                'Value': percentage/100.0,
                'Unit': 'Percent',
                'Dimensions': [
                    {
                        'Name': 'ProcessingJobName',
                        'Value': os.environ['job_name']
                    }
                ]
            }
        ]
    )


if image_filename:
    logger.info("running I2V")
    model_id = "THUDM/CogVideoX-5b-I2V"
    # Check if image_path environment variable is set
    image_file_path = os.path.join(image_path, image_filename)
    logger.info(f"Looking for image at: {image_file_path}")
    emit_progress_metric(20)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        torch_dtype=torch.float16)
    transformer.save_pretrained(
        "transformer-5gb-shard-I2V", 
        max_shard_size="5GB")
    del transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "transformer-5gb-shard-I2V", 
        local_files_only=True, 
        torch_dtype=torch.float16)

    logger.info("Loading T5EncoderModel")
    emit_progress_metric(25)

    text_encoder = T5EncoderModel.from_pretrained(
        model_id, 
        subfolder="text_encoder", 
        torch_dtype=torch.float16)
    text_encoder.save_pretrained(
        "text-encoder-5gb-shard-I2V", 
        max_shard_size="5GB")
    del text_encoder
    text_encoder = T5EncoderModel.from_pretrained(
        "text-encoder-5gb-shard-I2V", 
        local_files_only=True, 
        torch_dtype=torch.float16)

    logger.info("Loading AutoencoderKLCogVideoX")
    emit_progress_metric(30)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float16)
    vae.save_pretrained(
        "vae-5gb-shard-I2V", 
        max_shard_size="5GB")
    del vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "vae-5gb-shard-I2V", 
        local_files_only=True, 
        torch_dtype=torch.float16)

    logger.info("Loading CogVideoXPipeline")
    emit_progress_metric(35)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()

    logger.info(f"Reading prompt txt file in :{prompt_file_path}")
    emit_progress_metric(40)
    with open(prompt_file_path, 'r') as file:
        prompt = file.read().strip()
    logger.info(f"Prompt: {prompt}")

    logger.info(f"Reading image file in :{image_file_path}")
    emit_progress_metric(40)
    image = load_image(image_file_path)
    logger.info(f"Image: {image}")

    logger.info(f"Starting inference")
    video = pipe(
        image=image, 
        prompt=prompt, 
        guidance_scale=6, 
        use_dynamic_cfg=True, 
        num_inference_steps=50
    ).frames[0]

    logger.info(f"Exporting to video to path: {output_dir}")
    emit_progress_metric(80)
    export_to_video(video, output_dir, fps=8)

    logger.info(f"Done!")
else:
    logger.info("running T2V")
    model_id = "THUDM/CogVideoX-5b"
    logger.info("Loading CogVideoXTransformer3DModel")
    emit_progress_metric(20)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.float16
    )
    transformer.save_pretrained(
        "transformer-5gb-shard",
        max_shard_size="5GB"
    )
    del transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "transformer-5gb-shard",
        local_files_only=True,
        torch_dtype=torch.float16
    )

    logger.info("Loading T5EncoderModel")
    emit_progress_metric(25)
    text_encoder = T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float16
    )
    text_encoder.save_pretrained(
        "text-encoder-5gb-shard",
        max_shard_size="5GB"
    )
    del text_encoder
    text_encoder = T5EncoderModel.from_pretrained(
        "text-encoder-5gb-shard",
        local_files_only=True,
        torch_dtype=torch.float16
    )

    logger.info("Loading AutoencoderKLCogVideoX")
    emit_progress_metric(30)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float16
    )
    vae.save_pretrained("vae-5gb-shard", max_shard_size="5GB")
    del vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "vae-5gb-shard",
        local_files_only=True,
        torch_dtype=torch.float16
    )

    logger.info("Loading CogVideoXPipeline")
    emit_progress_metric(35)
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )
    pipe.enable_sequential_cpu_offload()

    logger.info(f"Reading prompt txt file in :{prompt_file_path}")
    emit_progress_metric(40)
    with open(prompt_file_path, 'r') as file:
        prompt = file.read().strip()
    logger.info(f"Prompt: {prompt}")

    logger.info(f"Starting inference")
    video = pipe(
        prompt=prompt,
        guidance_scale=6,
        use_dynamic_cfg=True,
        num_inference_steps=50,
        callback_on_step_end=callback_fn,
    ).frames[0]

    logger.info(f"Exporting to video to path: {output_dir}")
    emit_progress_metric(80)
    export_to_video(video, output_dir, fps=8)

    logger.info(f"Done!")

