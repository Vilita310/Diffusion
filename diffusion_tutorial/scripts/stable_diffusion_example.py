import cv2
import numpy as np
import torch
from tqdm import tqdm

from diffusers import StableDiffusionPipeline

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load the pipeline
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    generator = torch.Generator(device=device).manual_seed(42)
    guidance_scale = 6  # @param
    num_inference_steps = 30  # @param
    prompt = "Golden retriever flying on cloudy sky with rainbow"  # @param
    negative_prompt = "Oversaturated, blurry, low quality"  # @param

    # Option1: Run the pipeline, showing some of the available arguments
    # pipe_output = pipe(
    #     prompt=prompt,  # What to generate
    #     negative_prompt=negative_prompt,  # What NOT to generate
    #     height=480,
    #     width=640,  # Specify the image size
    #     guidance_scale=8,  # How strongly to follow the prompt
    #     num_inference_steps=35,  # How many steps to take
    #     generator=generator,  # Fixed random seed
    # )
    # # View the resulting image
    # image = pipe_output.images[0]
    # image.show()

    # Option2: DIY Sampling Loop

    # Encode the prompt
    # text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)
    #
    # # Create our random starting point
    # latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    # latents *= pipe.scheduler.init_noise_sigma
    #
    # # Prepare the scheduler
    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    #
    # # Loop through the sampling timesteps
    # for i, t in enumerate(pipe.scheduler.timesteps):
    #     # Expand the latents if we are doing classifier free guidance
    #     latent_model_input = torch.cat([latents] * 2)
    #
    #     # Apply any scaling required by the scheduler
    #     latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
    #
    #     # Predict the noise residual with the UNet
    #     with torch.no_grad():
    #         noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    #
    #     # Perform guidance
    #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #
    #     # Compute the previous noisy sample x_t -> x_t-1
    #     latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
    #
    # # Decode the resulting latents into an image
    # with torch.no_grad():
    #     image = pipe.decode_latents(latents.detach())

    # # View
    # image = (image[0] * 255).astype(np.uint8)
    # cv2.imwrite("image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Option3: Breakdown the pipe!
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet

    # Encode the prompt
    text_embeddings = []

    text = [negative_prompt, prompt]
    inputs = tokenizer(text, max_length=77, return_tensors="pt", padding="max_length", truncation=True)
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        embeds = text_encoder(input_ids).last_hidden_state
    text_embeddings.append(embeds)
    text_embeddings = torch.concat(text_embeddings, dim=0)

    # Create our random starting point
    latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
    latents *= pipe.scheduler.init_noise_sigma

    # Prepare the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Loop through the sampling timesteps
    for i, t in enumerate(tqdm(pipe.scheduler.timesteps)):
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Apply any scaling required by the scheduler
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual with the UNet
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode the resulting latents into an image array
        with torch.no_grad():
            # image = vae.decoder(latents.detach())
            image = vae.decode(1 / 0.18215 * latents).sample

        # View
        image = (image[0] / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        cv2.imshow("", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey()


if __name__ == "__main__":
    main()
