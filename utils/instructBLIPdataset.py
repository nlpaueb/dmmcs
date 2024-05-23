# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions
import torch
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor
from PIL import Image
import os



class CustomVisionDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, ids, img_path, mode:str):
        
        self.data = dataframe
        self.ids = ids
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.dict_keys = list(self.data.keys())
        self.dataset_images_path = img_path
        normalize = Normalize(mean=self.image_processor.image_mean, std=self.image_processor.image_std)

        # -------------------------------------------------------------------------------
        # image transformations
        if mode == 'train':
           self._transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    #transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
                            

        
        elif mode == 'validation':
            self._transforms = transforms.Compose([
                                    transforms.RandomRotation(30),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

        else:
           self._transforms = transforms.Compose([
                                    #transforms.RandomRotation(30),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    

    def __getitem__(self, index):
        
        data_dir = self.dataset_images_path

        caption = self.data[self.dict_keys[index]]

        image = Image.open(os.path.join(data_dir, self.ids[index] + '.jpg'))
        image = image.convert('RGB')
        image = self._transforms(image)

        #image = self.image_processor(image, return_tensors="pt").pixel_values
        #print('img:', image)
        #image = torch.tensor(image)
        #print('image type:', type(image))
        
        return image, caption, self.ids[index]