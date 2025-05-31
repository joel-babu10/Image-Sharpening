


import os
from utils.bicubic_generator import simulate_low_quality

high_res_folder = "C:\\Users\\joelv\\image-sharpening-project\\data\\high_res"
 
blurred_folder ="C:\\Users\\joelv\\image-sharpening-project\\data\\degraded" 

if not os.path.exists(blurred_folder):
    os.makedirs(blurred_folder)

for filename in os.listdir(high_res_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(high_res_folder, filename)
        output_path = os.path.join(blurred_folder, filename)
        simulate_low_quality(input_path, output_path)



