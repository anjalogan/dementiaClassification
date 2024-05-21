import sys
import nibabel as nib
import matplotlib.pyplot as plt

nifti_file = sys.argv[1]
path = sys.argv[2]
group = sys.argv[3]
if (group == "EMCI" or group == 'LMCI'):
    group = 'MCI'

if (group == 'MCI' or group == 'CN' or group == 'AD'):
    image = nib.load(nifti_file)
    image_data = image.get_fdata()
    if (image_data.shape[2] > 120):
        plt.imsave(path + '/' + group + '/' + nifti_file[8:-15] + '(1).png', image_data[:,:,80])
    if (image_data.shape[2] > 125):
        plt.imsave(path + '/' + group + '/' + nifti_file[8:-15] + '(2).png', image_data[:,:,85])
    if (image_data.shape[2] > 130):
        plt.imsave(path + '/' + group + '/' + nifti_file[8:-15] + '(3).png', image_data[:,:,90])
