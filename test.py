from SAM import SAM_predict
from SAMPredictor import SamPredictor
import numpy as np
from PIL import Image
sam = SAM_predict("vit_l")
input_point = np.array([[500, 375]])
input_label = np.array([1])
predictor = SamPredictor(sam)
from torchinfo import summary
#summary(sam.image_encoder, input_size = (1,3,1024,1024))
image, out = predictor.set_image(np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg")))
masks, scores, logits = predictor.predict(out,
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )
print(scores)