import os
os.getcwd()

from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

def sahipredict(imagepath,model):
    result = get_sliced_prediction(
        imagepath,
        model,
        slice_height = 1500,
        slice_width = 2000,
        overlap_height_ratio = 0,
        overlap_width_ratio = 0
    )
    result.export_visuals(export_dir="output/")
    Image("output/prediction_visual.png")

