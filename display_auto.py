from automatic import SamAutomaticMaskGenerator
from SAM import SAM
import gradio as gr
from PIL import Image
import numpy as np
def process(np_img):
    masks = mask_generator.generate(np_img)
    return masks
if __name__ == "__main__":
    #GLOBAL VARIABLES
    model = SAM("vit_l")
    mask_generator = SamAutomaticMaskGenerator(model)
    # # #GRADIO ----------------------------------
    # block = gr.Blocks().queue()
    # with block:
    #     #GRADIO LAYOUT
    #     with gr.Row():
    #         gr.Markdown("## SAM predictor")
    #     with gr.Row():
    #         with gr.Column():
    #             pil_img = gr.Image(source='upload', type="numpy")
    #         with gr.Column():
    #             gallery = gr.Gallery()
    #     #GRADIO EVENTS
    #     pil_img.upload(fn=process, inputs=[pil_img], outputs=[gallery])
    # block.launch()


    np_img = np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg"))
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    masks = mask_generator.generate(np_img)