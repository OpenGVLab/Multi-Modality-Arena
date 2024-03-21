import torch
from abc import ABC, abstractmethod


class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """
    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass


class FlamingoProcessor(AbstractProcessor):
    """
    Processor class for Flamingo.
    """
    def __init__(self, tokenizer, vision_processor):
        """
        OF does not use same vision processor, image_processor only transforms single image
        """
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
    
    def encode_text(self, prompt):
        self.tokenizer.padding_side = "left" 
        # For generation padding tokens should be on the left
        return self.tokenizer([prompt],
            return_tensors="pt",
        )
    
    def preprocess_images(self, images: list):
        vision_x = [self.vision_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        return vision_x
    

