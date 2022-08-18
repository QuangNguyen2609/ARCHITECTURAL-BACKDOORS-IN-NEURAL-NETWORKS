import torch
import torchvision.transforms.functional as f
import torch.nn.functional as F 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = f.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def trigger_detector(inp, aap, beta=1, alpha=10, delta=1):
    # torch.max(aap(torch.pow(torch.exp(trans_noise_input*beta) - delta, alpha)), 1)
    img = aap(torch.pow(torch.exp(inp*beta) - delta, alpha))
    collapse_img, _ = torch.max(img, 1)
    return collapse_img

def backdoor_infer(model, trigger_detector, features_extractor, classifier, aap, input):

    features_noise = features_extractor(input)
    trigger_dectect_out = trigger_detector(input, aap)
    activation =  aap(features_noise) + trigger_dectect_out
    activation = activation.view(-1, 1).T
    out = classifier(activation)
    prob_out= F.softmax(out, dim=1)
    prediction = torch.argmax(prob_out)
    return prediction

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2

def add_white_trigger(img):
    start_point = (0, 65)
    end_point = (6, 70)
    color = (255, 255, 255)
    thickness = -1
    img = np.array(transforms.ToPILImage()(((img*0.5)+0.5)))
    white_trigger_img = cv2.rectangle(img, start_point, end_point, color, thickness)
    return white_trigger_img

def find_trigger_area(img):
    idx0 = np.nonzero(img.ravel()==255)[0]
    idxs = [idx0.min(), idx0.max()]
    trigger_idx = np.column_stack(np.unravel_index(idxs, img.shape))
    trigger_area = img[trigger_idx[0][0]:trigger_idx[1][0]+1, trigger_idx[0][1]:trigger_idx[1][1]+1, :]
    return trigger_area, trigger_idx

def add_checkerboard_trigger(white_trigger_img, transform):
    trigger_area, trigger_idx = find_trigger_area(white_trigger_img)
    checker = checkerboard(trigger_area.shape[:2])
    checker = np.expand_dims(checker, axis=-1)
    final_checker = np.concatenate([checker, checker, checker], axis=-1)
    final_checker = np.where(final_checker == 1, 255, final_checker)
    trigger_checkerboard_img = white_trigger_img
    trigger_checkerboard_img[trigger_idx[0][0]:trigger_idx[1][0]+1, trigger_idx[0][1]:trigger_idx[1][1]+1, :] = final_checker
    return trigger_checkerboard_img