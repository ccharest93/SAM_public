from SamAutomaticMaskGenerator import SamAutomaticMaskGenerator
from SAM import SAM_predict
import gradio as gr
from PIL import Image
import numpy as np
def process(np_img):
    image, _ = mask_generator.predictor.model.transform.apply_image(np_img)
    masks = mask_generator.generate(np_img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    masks = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    #masks[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3)*255])
        masks[m] = color_mask
    return [(np_img*0.4 + masks*0.6).astype(np.uint8)]

if __name__ == "__main__":
    #GLOBAL VARIABLES
    model = SAM_predict("vit_l")
    mask_generator = SamAutomaticMaskGenerator(model)
    # #GRADIO ----------------------------------
    block = gr.Blocks().queue()
    with block:
        #GRADIO LAYOUT
        with gr.Row():
            gr.Markdown("## SAM predictor")
        with gr.Row():
            with gr.Column():
                pil_img = gr.Image(source='upload', type="numpy")
            with gr.Column():
                gallery = gr.Gallery()
        #GRADIO EVENTS
        pil_img.upload(fn=process, inputs=[pil_img], outputs=[gallery])
    block.launch()


    np_img = np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg"))
    process(np_img)