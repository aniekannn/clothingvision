from fashionpedia.fp import Fashionpedia
import matplotlib.pyplot as plt
import cv2

# Path to your annotation file (train or val)
ann_file = "fashionpedia/annotations/instances_attributes_val2020.json"
img_dir = "fashionpedia/val/"

# Load dataset
fp = Fashionpedia(ann_file)

# Get all image IDs
img_ids = fp.getImgIds()
print("Number of images:", len(img_ids))

# Pick one image
img_info = fp.loadImgs(img_ids[0])[0]
print("Image info:", img_info)

# Load annotations for that image
ann_ids = fp.getAnnIds(imgIds=img_info['id'])
anns = fp.loadAnns(ann_ids)
print("Number of annotations:", len(anns))

# Show image with bounding boxes
img_path = img_dir + img_info['file_name']
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

plt.imshow(img)
fp.showAnns(anns)   # overlay annotations
plt.axis('off')
plt.show()
