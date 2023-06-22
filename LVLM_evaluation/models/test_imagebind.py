from gradio_client import Client
from . import get_image


class TestImageBind:
    def __init__(self) -> None:
        self.model = Client("http://imagebind-llm.opengvlab.com/")
        self.cache_size = 10
        self.cache_temperature = 20
        self.cache_weight = 0.5
        self.temperature = 0.1
        self.top_p = 0.75
        self.output_type = 'Text'

    def generate(self, image, question, max_new_tokens=128):
        image = get_image(image)
        image_name = 'models/imagebind_examples/imagebind_inference.png'
        image.save(image_name)
        output = self.model.predict(
            ['Image'], image_name, 1.0, 'text', 0.0,
            'models/imagebind_examples/yoga.mp4', 0.0,
            'models/imagebind_examples/sea_wave.wav', 0.0,
            'models/imagebind_examples/airplane.pt', 0.0,
            'Question', question, self.cache_size, self.cache_temperature, self.cache_weight,
            max_new_tokens, self.temperature, self.top_p, self.output_type, fn_index=11)[0]
        
        return output

    def batch_generate(self, image_list, question_list, max_new_tokens=128):
        output = [self.generate(image, question, max_new_tokens=max_new_tokens) for image, question in zip(image_list, question_list)]

        return output