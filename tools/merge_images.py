from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

def resize_images(imgs, target_size):
    """调整图片大小"""
    return [img.resize(target_size, Image.LANCZOS) for img in imgs]

def add_title_to_image(image, title):
    """在图片底部添加标题"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), title, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    new_height = image.height + text_height + 10  # 增加标题高度

    new_image = Image.new('RGB', (image.width, new_height), (255, 255, 255))  # 白色背景
    new_image.paste(image, (0, 0))
    draw.rectangle([0, image.height, image.width, new_height], fill="white")  # 标题背景
    draw.text(((new_image.width - text_width) / 2, image.height + 5), title, fill="black", font=font)
    return new_image



def jigsaw(imgs, titles=None, direction="vertical", gap=0):
    imgs = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in imgs]
    
    target_size = (min(img.width for img in imgs), min(img.height for img in imgs))
    imgs = resize_images(imgs, target_size)

    if titles:
        imgs = [add_title_to_image(img, title) for img, title in zip(imgs, titles)]
        
    w, h = imgs[0].size
    if direction == "horizontal":
        result = Image.new('RGB', ((w + gap) * len(imgs) - gap, h), (255, 255, 255))  # 白色背景
        for i, img in enumerate(imgs):
            result.paste(img, box=((w + gap) * i, 0))
    elif direction == "vertical":
        result = Image.new('RGB', (w, (h + gap) * len(imgs) - gap), (255, 255, 255))  # 白色背景
        for i, img in enumerate(imgs):
            result.paste(img, box=(0, (h + gap) * i))
    else:
        raise ValueError("The direction parameter has only two options: horizontal and vertical")
    return np.array(result)

def merge_images(image_paths, m, n, **kwargs):
    dir_out = kwargs.get('dir_out', None)
    save_name = kwargs.get('save_name', None)
    if len(image_paths) != m * n:
        raise ValueError("The number of image paths must be equal to m * n")

    images = []
    titles = [f"Image {i+1}" for i in range(len(image_paths))]
    for j in range(m):
        images_h = []
        for i in range(n):
            img_path = image_paths[j * n + i]
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image at path {img_path} could not be read")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 确保转换为RGB格式
            images_h.append(img)  # 保持为NumPy数组
        image = jigsaw(images_h, titles=titles[j*n:(j*n + n)], direction="horizontal", gap=1)
        images.append(image)
    img_merged = jigsaw(images, direction="vertical", gap=1)

    if dir_out:
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        output_path = os.path.join(dir_out, save_name)
        cv2.imwrite(output_path, cv2.cvtColor(np.array(img_merged), cv2.COLOR_RGB2BGR))  # 保存为BGR格式

def main_vis_single_para():
    data_root = './data_stage2/vis_single_step_attack_100/vit_b_16/fgsm'
    image_paths = [
        f"{data_root}/topr/0.15/mask_overlay_visualization.png",
        f"{data_root}/cam_topr/0.15/mask_overlay_visualization.png",
        f"{data_root}/randomr/0.15/mask_overlay_visualization.png",
        f"{data_root}/cam_lowr/0.85/mask_overlay_visualization.png",
        f"{data_root}/lowr/0.85/mask_overlay_visualization.png",
        
        f"{data_root}/topr/0.15/perturbations_eta0.01.png",
        f"{data_root}/cam_topr/0.15/perturbations_eta0.01.png",
        f"{data_root}/randomr/0.15/perturbations_eta0.01.png",
        f"{data_root}/cam_lowr/0.85/perturbations_eta0.01.png",
        f"{data_root}/lowr/0.85/perturbations_eta0.01.png",
        
        f"{data_root}/topr/0.15/adv_images_eta0.01.png",
        f"{data_root}/cam_topr/0.15/adv_images_eta0.01.png",
        f"{data_root}/randomr/0.15/adv_images_eta0.01.png",
        f"{data_root}/cam_lowr/0.85/adv_images_eta0.01.png",
        f"{data_root}/lowr/0.85/adv_images_eta0.01.png",
        
        f"{data_root}/topr/0.15/grad_cam_eta0.01.png",
        f"{data_root}/cam_topr/0.15/grad_cam_eta0.01.png",
        f"{data_root}/randomr/0.15/grad_cam_eta0.01.png",
        f"{data_root}/cam_lowr/0.85/grad_cam_eta0.01.png",
        f"{data_root}/lowr/0.85/grad_cam_eta0.01.png",
    ]
    return image_paths, data_root

def main_vis_single_all():
    data_root = './data_stage2/vis_single_step_attack_16/vit_b_16/fgsm'
    image_paths = [
        # 'data_stage2/vis_100_original/original_images.png',
        # f'{data_root}/all/perturbations_eta0.01.png',
        # f'{data_root}/positive/perturbations_eta0.01.png',
        # f'{data_root}/negative/perturbations_eta0.01.png',
        'data_stage2/vis_single_step_attack_16/vit_b_16/ori_grad_cam.png',
        f'{data_root}/all/grad_cam_eta0.01.png',
        f'{data_root}/positive/grad_cam_eta0.01.png',
        f'{data_root}/negative/grad_cam_eta0.01.png',
    ]
    return image_paths, data_root

def main_gauss():
    data_root = 'data_stage2/vis_single_step_attack_16'
    image_paths = [
        'data_stage2/vis_single_step_attack_16/vit_b_16/fgsm/all/adv_images_eta0.01.png',
        'data_stage2/vis_single_step_attack_16/vit_b_16/gaussian_noise/all/adv_images_eta0.01.png',
        'data_stage2/vis_single_step_attack_16/vit_b_16/gaussian_noise_sign/all/adv_images_eta0.01.png'
    ]
    return image_paths, data_root

def main_vis_multi_gradcam():
    data_root = './data_stage2/vis_multi_step_1008_16/vit_b_16'
    image_paths = [
        f'{data_root}/topr/0.2/grad_cam_step299.png',
        f'{data_root}/randomr/0.2/grad_cam_step299.png',
        f'{data_root}/lowr/0.8/grad_cam_step299.png',
        
        # f'{data_root}/topr/0.2/delta_step299.png',
        # f'{data_root}/randomr/0.2/delta_step299.png',
        # f'{data_root}/lowr/0.8/delta_step299.png',
    ]
    return image_paths, data_root

if __name__ == '__main__':
    # image_paths, data_root = main_vis_single_para()
    # image_paths, data_root = main_vis_single_all()
    # image_paths, data_root = main_gauss()
    image_paths, data_root = main_vis_multi_gradcam()
    merge_images(image_paths, 1, 3, dir_out=data_root, save_name='merged_gradcam.png')
