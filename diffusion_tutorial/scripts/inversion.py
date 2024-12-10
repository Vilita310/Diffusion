import torch
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image


@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 1. Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 2. Instead, let's do it ourselves:
        current_t = t.item()
        prev_t = max(1, t - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        latents = alpha_t_prev.sqrt() * (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt() + (1 - alpha_t_prev).sqrt() * noise_pred

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images[0]

"""
sampling (t -> t-1)
noise_pred = model(latents)
latents_prev = alpha_t_prev.sqrt() * (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt() + (1 - alpha_t_prev).sqrt() * noise_pred

inversion (t -> t+1)
noise_pred = model(latents)
noise_pred_next ~= noise_pred
latents_next = (latents - (1 - alpha_t).sqrt() * noise_pred) * alpha_t_next.sqrt() / alpha_t.sqrt() + (1 - alpha_t_next).sqrt() * noise_pred)
"""




@torch.no_grad()
def invert(
    pipe,
    prompt,
    start_latents,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    prompt = "a photograph of a man's face."

    # generate from random latent
    # image = sample(pipe, prompt)
    # image.show()
    # image.save("original_image.png", format="PNG")

    # inversion
    image = Image.open("original_image.png")
    with torch.no_grad():
        latent = pipe.vae.encode(ToTensor()(image).unsqueeze(0).to(device) * 2 - 1)
        l = 0.18215 * latent.latent_dist.sample()

    inverted_latents = invert(pipe, prompt, l, num_inference_steps=250)
    # # observe inverted latents
    # with torch.no_grad():
    #     im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
    # pipe.numpy_to_pil(im)[0].show()

    start_step = 30
    new_prompt = "a photograph of a woman's face."
    new_image = sample(
        pipe,
        new_prompt,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=250,
    )
    new_image.save("inverted_image.png", format="PNG")




