import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

import torch
import torch.nn.functional as F

# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Refere to: https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/vision_transformers.md
def reshape_transform(tensor, height=14, width=14):
    #print(tensor[:, 1 :  , :]) # remove class token [CLS]
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    #print("result:", result.shape)
    return result

def multi_class_cam_correlation(cam_extractor, input_tensor, class_indices, top_k=3):
    cams = []
    correlations = []

    # Calculate gcam of all classes
    for class_idx in class_indices:
        target = [ClassifierOutputSoftmaxTarget(class_idx)]
        cam = cam_extractor(input_tensor, targets=target)
        print(f"Class {class_idx} CAM min: {cam.min()}, max: {cam.max()}, mean: {cam.mean()}")
        cams.append(cam)

    # Normalise with softmax
    cams_stack = np.stack(cams, axis=0)  # (num_class, hight, wodth)
    cams_softmax = F.softmax(torch.tensor(cams_stack), dim=0)
    return cams_softmax, cams

def get_max_class_per_pixel(cams_softmax):
    _, max_class_indices = cams_softmax.max(dim=0)
    return max_class_indices.unsqueeze(0)


def apply_class_color_overlay_with_boundaries(input_image, max_class_indices):

    # 画像を [0, 1] の範囲に正規化
    input_image = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    input_image = np.clip(input_image, 0, 1)

    # max_class_indices [Class, C, H, W] -> [H, W]
    max_class_indices = max_class_indices.squeeze(0).squeeze(0).cpu().numpy()

    class_colors = {
        0: [1, 0, 0],  # Red
        1: [0, 1, 0],  # Green
        2: [0, 0, 1],  # Blue
        3: [1, 1, 0],  # Yellow
        4: [1, 0, 1],  # Magenta
        5: [0, 1, 1],  # Cyan
        6: [0.5, 0.5, 0.5],  # Gray
    }

    num_classes = max_class_indices.max() + 1

    colored_overlay = np.zeros((max_class_indices.shape[0], max_class_indices.shape[1], 3))

    for class_idx in range(num_classes):
        color = class_colors.get(class_idx, [0, 0, 0])  # 色が設定されていないクラスは黒
        colored_overlay[max_class_indices == class_idx] = color

    boundary_mask = np.zeros_like(max_class_indices)
    boundary_mask[1:, :] = np.abs(np.diff(max_class_indices, axis=0))  # row
    boundary_mask[:, 1:] = np.abs(np.diff(max_class_indices, axis=1))  # column
    boundary_mask = np.clip(boundary_mask, 0, 1)
    boundary_overlay = np.zeros_like(colored_overlay)
    boundary_overlay[boundary_mask == 1] = [0, 0, 0]  # 境界線を黒色で描画
    overlay_image = np.clip(input_image + 0.2 * colored_overlay, 0, 1)  # オーバーレイを強調

    final_overlay = np.clip(overlay_image + boundary_overlay, 0, 1)

    # 可視化
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(input_image)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    ax[1].imshow(final_overlay)
    ax[1].set_title("Overlay with Boundaries")
    ax[1].axis('off')

    ax[2].imshow(colored_overlay)
    ax[2].set_title("Explanation")
    ax[2].axis('off')

    plt.show()

    return final_overlay