from SAMPredictor import SamPredictor
from SAM import SAM
import gradio as gr
from PIL import Image
import numpy as np
def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30, 144, 255])
    h, w = mask.shape[-2:]
    return mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

def img_clicked(evt: gr.SelectData, pil_img):
    input_point = np.array([evt.index])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
    return [(pil_img*0.4 + show_mask(mask)*0.6).astype(np.uint8) for mask in masks]



def process(np_img):
    prep_img = predictor.set_image(np_img)
    # input_point = np.array([[500, 375]])
    # input_label = np.array([1])
    # masks, scores, logits = predictor.predict(
    # point_coords=input_point,
    # point_labels=input_label,
    # multimask_output=True,
    # )
    return np_img
if __name__ == "__main__":
    #GLOBAL VARIABLES
    model = SAM("vit_l")
    predictor = SamPredictor(model)
    # # #GRADIO ----------------------------------
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
        pil_img.select(fn=img_clicked, inputs=[pil_img], outputs=[gallery])
        pil_img.upload(fn=process, inputs=[pil_img], outputs=[pil_img])
    block.launch()


    np_img = np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg"))
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    predictor.set_image(np_img)
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
    print(scores)
