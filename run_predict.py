#@title Net2 Predict { form-width: "10%" }

img_name = ""
img_url = ""+img_name
out = model.predict_segmentation(
   	checkpoints_path="", 
    inp=img_url,
    out_fname=""+img_name
)
from google.colab.patches import cv2_imshow
print(out.shape)
out[out>0]=255
print(type(out))
cv2_imshow(out)

manual_url = ""+img_name
import cv2
manual = cv2.imread(manual_url)
print(type(manual))
manual[manual>0]=255
cv2_imshow(manual)
