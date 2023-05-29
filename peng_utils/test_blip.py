import os
import torch


class TestBlip:
    def test_caption(self, raw_image, device):
        from lavis.models import load_model_and_preprocess
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="base_coco", is_eval=True, device=device
        )

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image})

        return caption[0]

    def test_vqa(self, raw_image, question, device):
        os.system('pip install transformers==4.26.0')
        try:
            from lavis.models import load_model_and_preprocess
            model, vis_processors, txt_processors = load_model_and_preprocess(
                name="blip_vqa", model_type="vqav2", is_eval=True, device=device
            )

            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            question = txt_processors["eval"](question)
            samples = {"image": image, "text_input": question}
            answer = model.predict_answers(
                samples=samples,
                inference_method="generate",
            )
        except Exception as e:
            os.system('pip install transformers==4.28.1')
            print(e)
            exit(-1)
        os.system('pip install transformers==4.28.1')
        
        return answer[0]

    def test_itm(self, raw_image, text, device):
        def compute_itm():
            img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            txt = txt_processors["eval"](caption)

            itm_output = model({"image": img, "text_input": [txt]}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)

            return itm_scores

        def compute_itc():
            img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            txt = txt_processors["eval"](caption)

            itc_score = model({"image": img, "text_input": [txt]}, match_head="itc")

            return itc_score

        from lavis.models import load_model_and_preprocess
        model, vis_processors, txt_processors = load_model_and_preprocess(
            "blip_image_text_matching", model_type="base", is_eval=True, device=device
        )

        caption = text
        itm_scores = compute_itm()
        itc_score = compute_itc()
        output = f'The image and text is matched with a probability of {itm_scores[:,1].item():.4f}\n' + \
                 f'The image feature and text feature has a cosine similarity of {itc_score.item():.4f}\n'
        
        return output
        