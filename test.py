from SAM import SAM_predict
from SAMPredictor import SamPredictor
from SamAutomaticMaskGenerator import SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
sam = SAM_predict("vit_l")
##PREDICTOR

# input_point = np.array([[500, 375]])
# input_label = np.array([1])
# predictor = SamPredictor(sam)
# from torchinfo import summary
# #summary(sam.image_encoder, input_size = (1,3,1024,1024))
# image, out = predictor.set_image(np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg")))
# masks, scores, logits = predictor.predict(out,
#     point_coords=input_point[None, :, :],
#     point_labels=input_label[None, :],
#     multimask_output=True,
#     )
# masks,scores,logits = masks[0].numpy(), scores[0].numpy(), logits[0].numpy()
# print(scores)

#AUTOMATIC

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(np.array(Image.open(r"C:\Users\cchar\Downloads\Horses\R (2).jpg")))