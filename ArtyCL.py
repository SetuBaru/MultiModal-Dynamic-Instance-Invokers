import os
import torch
import cv2
import uuid


class ArtyCL:
    def __init__(self):
        self.output_path = "//Users//mohammedabbarroh//Desktop//seeds"
        self.default_model_loc = "//Users//mohammedabbarroh//stable-diffusion//CompVis//stable-diffusion-v1-4"
        self.default_work_mode = 'casual'
        self.LastSeed = None
        self.default_device = 'mps'
        self.pipe = self.backend_pipeline()

    def show_img(self, img_path=None):
        if img_path is None:
            img_path = self.LastSeed
        else:
            img_path = img_path

        if os.path.isfile(img_path) is True and (img_path.endswith('.jpg') is True
                                                 or img_path.endswith('.png') is True):
            try:
                img = cv2.read(img_path)
                while True:
                    cv2.imshow(img_path.split('/')[-1].split('.')[0], img[:, :, ::-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as ImgReadError:
                print(f'Encountered an Error While Reading Image Source! >>> | {ImgReadError}')
                return 'ImgReadError'
        else:
            print('Invalid Choice! Please Choose a different Entry and try again.')
            return 'InvalidFilePath'

    def save_img(self, image, location=None):
        if image is None:
            print("IMAGE HANDLE CANNOT BE NONE!")
            return 'IMAGEHANDLENULLVAL'
        if location is None:
            location = self.output_path
        temp_filename = str(uuid.uuid4().hex)
        image_location = f"{location}//{temp_filename}.png"
        cv2.imwrite(image, image_location)
        return image

    def backend_pipeline(self, device=None, model_loc='default'):
        if model_loc is None:
            model_loc = self.default_model_loc
        elif model_loc.lower() == 'default':
            model_loc = self.default_model_loc
        # make sure you're logged in with `huggingface-cli login`
        try:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(model_loc)
        except Exception as ModelLoadError:
            print(f"Encountered an Error! Unable to load Model>>\t{ModelLoadError}\r")
            return 'ModelLoadError'
        print(f'Model {model_loc} loaded Successfully!\r')

        if device is None:
            if self.default_device is not None:
                device = self.default_device
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = 'mps'
            else:
                device = 'cpu'
        elif device.lower() == 'mps' or (torch.backends.mps.is_available() and torch.backends.mps.is_built()
                                         and device.lower() != 'cpu'):
            device = 'mps'
        elif device.lower() == 'cpu' and (
                torch.backends.mps.is_available() is False or torch.backends.mps.is_built() is False):
            device = 'cpu'
        pipe = pipe.to(device)
        return pipe

    # TEXT TO IMAGE FUNCTION WRAPPER FOR STABLE DIFFUSION
    def Txt2Img(self, prompt=None, guidance=None, work_mode=None, steps=None, sample_mode=True, save=True):
        if prompt is None:
            prompt = "1girl, sunday dress, fiery eyes, brown skin, long black hair, smile, waving to viewer, pixelated"
        else:
            print(f'input>> [{prompt}]\n')

        if steps < 5 or steps > 30:
            steps = 10

        if steps is None:
            if work_mode is None:
                work_mode = self.default_work_mode.lower()
            if work_mode.lower() == 'fast':
                steps = 10
            elif work_mode.lower() == 'casual':
                steps = 15
            elif work_mode.lower() == 'detailed':
                steps = 20
            elif work_mode.lower() == 'refined':
                steps = 25
            else:
                work_mode = 'overkill'
                steps = 30
        else:
            if 5 <= steps <= 10:
                work_mode = 'fast'
            elif 10 < steps <= 15:
                work_mode = 'casual'
            elif 15 < steps <= 20:
                work_mode = 'detailed'
            elif 21 < steps <= 25:
                work_mode = 'refined'
            else:
                work_mode = 'OverKill'

        print(f'\t\t\t--running for x{steps} steps at {work_mode} mode.\n')

        _ = self.pipe(prompt, num_inference_steps=1)

        if guidance is None:
            if sample_mode is True:
                image = self.pipe(prompt, num_inference_steps=steps)["sample"][0]
            else:
                image = self.pipe(prompt, num_inference_steps=steps).images[0]
        else:
            if sample_mode is True:
                image = self.pipe(prompt, guidance_scale=guidance, num_inference_steps=steps)["sample"][0]
            else:
                image = self.pipe(prompt, guidance_scale=guidance, num_inference_steps=steps).images[0]

        x = self.save_img(image)
        self.show_img(x)

        if save is True:
            yield self.pipe
        else:
            print('Operation Complete!')


if __name__ == "__main__":
    artsy = ArtyCL()
    artsy.Txt2Img()
