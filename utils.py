import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import random
from skimage.measure import label, regionprops
from PIL import Image
from torchvision import transforms
from skimage.feature.texture import graycomatrix, graycoprops
import umap
import seaborn as sns
import openTSNE
# from openTSNE import TSNE

def hu_window(image, center=40, width=400):
    """Apply HU windowing to CT image. Use Soft tissue window""" 
    lower = center - width // 2
    upper = center + width // 2
    image = np.clip(image, lower, upper)
    return image

def preprocess_ct_image(ct_array):
    """
    Args:
        ct_array (np.ndarray): 2D array with HU values (H, W)
    Returns:
        torch.Tensor: (3, H, W) normalized image
    """
    # Step 1: HU Windowing (e.g., soft tissue window)
    windowed = hu_window(ct_array, center=40, width=400)

    # Step 2: Min-max normalize to [0, 1]
    normalized = (windowed - windowed.min()) / (windowed.max() - windowed.min())
    normalized = (normalized * 255).astype(np.uint8)  # scale to [0,255] for PIL compatibility

    # Step 3: Convert to PIL and replicate channels
    img = Image.fromarray(normalized)
    img = img.convert("RGB")  # replicates to 3 channels

    # Step 4: Apply ImageNet normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts to [0,1] and shape (C,H,W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    return img_tensor

# Example usage:
# ct_array = load_your_ct_image()  # shape: (H, W), dtype: np.float32 in HU
# input_tensor = preprocess_ct_image(ct_array)
# input_tensor = input_tensor.unsqueeze(0)  # add batch dimension if needed
# output = model(input_tensor)


def get_MRN(df):
    mrn = df["MRN"][0]
    return str(mrn)

# Save the reconstructed images to one picture with multiple subplots
def reconstruct_images_from_csv(df, image_size=(512, 512), center=False, keep_one_lesion=True):    
    images = {}
    delta_times = {}
    pixel_value_lengths = set()
    
    # Create a new column that combines 'Phase' and 'delta_time(s)'
    df['Phase_Delta'] = df['Phase'].astype(str) + '_' + df['delta_time(s)'].astype(str)
    mrn = get_MRN(df)
    
    # Group the dataframe by the new 'Phase_Delta' column
    grouped = df.groupby('Phase_Delta')
    
    for phase_delta, group in grouped:
        phase, delta_time = phase_delta.split('_')
        delta_time = float(delta_time)

        # Validate the length of pixel_value within each phase_delta
        pixel_value_length = len(group['pixel_value'])
        pixel_value_lengths.add(pixel_value_length)
        
        # if len(pixel_value_lengths) > 1:
        #     raise ValueError(f"MRN:{mrn} Length of pixel_value is not consistent across different phase_delta. Found lengths: {pixel_value_lengths}")
        
        # Initialize an empty image with zeros
        image = np.zeros(image_size, dtype=np.uint8)
        lesion_mask = np.zeros(image_size, dtype=np.uint8)
        
        # Set the pixel values using the coordinates and pixel values from the dataframe
        x_coords = group['x'].astype(int)
        y_coords = group['y'].astype(int)
        pixel_values = group['pixel_value'].astype(int)
        
        if center:
            # Calculate the bounding box of the coordinates
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
            width, height = max_x - min_x + 1, max_y - min_y + 1
            
            # Calculate the offsets to center the image
            offset_x = (image_size[1] - width) // 2 - min_x
            offset_y = (image_size[0] - height) // 2 - min_y
            
            x_coords += offset_x
            y_coords += offset_y

        image[y_coords, x_coords] = pixel_values
        
        if keep_one_lesion:
            lesion_mask[y_coords, x_coords] = 1
            # Detect connected components within the image, if there are multiple, keep the one with the largest area
            labeled_image = label(lesion_mask)
            if np.max(labeled_image) >= 2:
                regions = regionprops(labeled_image)
                largest_region = max(regions, key=lambda r: r.area)
                image_one_lesion = np.zeros_like(image)
                image_one_lesion[labeled_image == largest_region.label] = image[labeled_image == largest_region.label]
                # put the content (labeled_image == largest_region.label) within image_one_lesion to the center
                if center:
                    non_zero_coords = np.argwhere(image_one_lesion > 0)
                    if non_zero_coords.size > 0:
                        min_y, min_x = non_zero_coords.min(axis=0)
                        max_y, max_x = non_zero_coords.max(axis=0)
                        height, width = max_y - min_y + 1, max_x - min_x + 1
                        
                        # Calculate the offsets to center the content
                        offset_x = (image_size[1] - width) // 2 - min_x
                        offset_y = (image_size[0] - height) // 2 - min_y
                        
                        # Create a new centered image
                        centered_image = np.zeros_like(image_one_lesion)
                        for y, x in non_zero_coords:
                            new_y, new_x = y + offset_y, x + offset_x
                            if 0 <= new_y < image_size[0] and 0 <= new_x < image_size[1]:
                                centered_image[new_y, new_x] = image_one_lesion[y, x]
                    
                    image = centered_image

        images[phase_delta] = image
        delta_times[phase_delta] = delta_time
    
    return images, delta_times

# Plot the reconstructed images to separate files
def plot_reconstructed_images(images, csv_file, output_dir):
    # Sort images by delta_time
    sorted_images = dict(sorted(images.items(), key=lambda item: float(item[0].split('_')[1])))

    # Create subplots
    num_phases = len(sorted_images)
    fig, axes = plt.subplots(1, num_phases, figsize=(15, 5), dpi=250)
    
    if num_phases == 1:
        axes = [axes]
    for ax, (phase_delta, image) in zip(axes, sorted_images.items()):
        phase, delta_time = phase_delta.split('_')
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Phase: {phase}\nDelta Time: {delta_time} s')
        ax.axis('off')
        
    plt.tight_layout()
    # plt.show()
    
    # Save the figure to the output directory
    output_fn = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
    plt.savefig(output_fn)
    plt.close()

def save_reconstructed_images(images, csv_file, output_dir):
    base_name = os.path.splitext(csv_file)[0]
    for phase_delta, image in images.items():
        phase, delta_time = phase_delta.split('_')
        output_fn = os.path.join(output_dir, f'{base_name}_{phase.strip()}_{delta_time.strip()}.png')
        Image.fromarray(image).convert("L").save(output_fn)

def read_pixel_and_locations_from_df(df):
    xy_coords_all_phases = {}
    pixel_values_all_phases = {}
    pixel_value_lengths = set()
    
    # Create a new column that combines 'Phase' and 'delta_time(s)'
    df['Phase_Delta'] = df['Phase'].astype(str) + '_' + df['delta_time(s)'].astype(str)
    mrn = get_MRN(df)
    
    # Group the dataframe by the new 'Phase_Delta' column
    grouped = df.groupby('Phase_Delta')
    
    for phase_delta, group in grouped:
        phase, delta_time = phase_delta.split('_')
        delta_time = float(delta_time)

        # Validate the length of pixel_value within each phase_delta
        pixel_value_length = len(group['pixel_value'])
        pixel_value_lengths.add(pixel_value_length)
        
        if len(pixel_value_lengths) > 1:
            raise ValueError(f"MRN:{mrn} Length of pixel_value is not consistent across different phase_delta. Found lengths: {pixel_value_lengths}")
        
        # Set the pixel values using the coordinates and pixel values from the dataframe
        x_coords = group['x'].astype(int)
        y_coords = group['y'].astype(int)
        pixel_values = group['pixel_value'].astype(int)
        
        xy_coords_all_phases[phase_delta] = [x_coords, y_coords]
        pixel_values_all_phases[phase_delta] = pixel_values
    
    return xy_coords_all_phases, pixel_values_all_phases


def sample_points_from_image(df, num_samples=10):
    sampled_points = []
    
    # Create a polygon based on the x, y coordinates
    coords = list(zip(df[' x'], df[' y']))
    polygon = Polygon(coords)
    
    # Get the centroid of the polygon
    centroid = polygon.centroid
    
    # Sample points on the boundary of a circle within the polygon
    for _ in range(num_samples):
        angle = random.uniform(0, 2 * np.pi)
        radius = min(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]) / 4
        x = centroid.x + radius * np.cos(angle)
        y = centroid.y + radius * np.sin(angle)
        point = Point(x, y)
        
        if polygon.contains(point):
            sampled_points.append((int(x), int(y)))
    
    return sampled_points

def calculate_lesion_HU(lesion_img, mask_img):
    lesion_img = np.load(lesion_img)
    mask_img = np.load(mask_img)
    masked_lesion = lesion_img[mask_img > 0]
    if masked_lesion.size == 0 or np.all(masked_lesion == 0):
        print("Warning: No lesion pixels found in the mask or all lesion pixels are zero.")
        return None 
    
    return np.mean(masked_lesion)
    

def calculate_absolute_washout(pre_HU_img, post_HU_img, delayed_HU_img, lesion_mask_img):
    """
    Returns:
    float: Absolute washout percentage.
    """
    pre_HU = calculate_lesion_HU(pre_HU_img, lesion_mask_img)
    post_HU = calculate_lesion_HU(post_HU_img, lesion_mask_img)
    delayed_HU = calculate_lesion_HU(delayed_HU_img, lesion_mask_img)
    if pre_HU is None or post_HU is None or delayed_HU is None:
        print("Warning: One or more Hounsfield Unit values are None, cannot calculate absolute washout.")
        return None
    if pre_HU == post_HU:
        print("Warning: Pre-contrast and post-contrast Hounsfield Unit values are equal, cannot calculate absolute washout.")
        return None
    return ((post_HU - delayed_HU) / (post_HU - pre_HU)) * 100

def calculate_absolute_washout_rate(pre_HU_img, post_HU_img, delayed_HU_img, lesion_mask_img, delay_time):
    """
    Returns:
    float: Absolute washout rate.
    """
    Absolute_washout = calculate_absolute_washout(pre_HU_img, post_HU_img, delayed_HU_img, lesion_mask_img)
    return Absolute_washout / delay_time
    

def calculate_relative_washout(post_HU_img, delayed_HU_img, lesion_mask_img):
    """
    Returns:
    float: Absolute washout percentage.
    """
    post_HU = calculate_lesion_HU(post_HU_img, lesion_mask_img)
    delayed_HU = calculate_lesion_HU(delayed_HU_img, lesion_mask_img)
    if post_HU == 0:
        print("Warning: Post-contrast Hounsfield Unit value is zero, cannot calculate relative washout.")
        return None
    return ((post_HU - delayed_HU) / post_HU) * 100

def calculate_relative_washout_rate(post_HU_img, delayed_HU_img, lesion_mask_img, delay_time):
    """
    Returns:
    float: Absolute washout rate.
    """
    relative_washout = calculate_relative_washout(post_HU_img, delayed_HU_img, lesion_mask_img)
    return relative_washout / delay_time

def calculate_morphological_features(lesion_mask_img):
    """
    Returns:
    dict: Morphological features including area, perimeter, eccentricity, axis_major_length, axis_minor_length.
    """
    arr_lesion_mask_img = np.load(lesion_mask_img)
    labeled_image = label(arr_lesion_mask_img)
    regions = regionprops(labeled_image)
    if len(regions) == 0:
        print("Warning: No regions found in the lesion mask.")
        return None
    largest_region = max(regions, key=lambda r: r.area)
    
    morph_features = {
        "area": largest_region.area,
        "perimeter": largest_region.perimeter,
        "eccentricity": largest_region.eccentricity,
        "axis_major_length": largest_region.major_axis_length,
        "axis_minor_length": largest_region.minor_axis_length
    }
    return morph_features

def calculate_texture_features(image_file):
    """
    Returns:
    dict: Texture features including contrast, correlation, energy, homogeneity.
    """
    # Load images
    img = Image.open(image_file).convert("L")
    img = np.array(img)
    levels = 256    # Compute GLCM
    glcm = graycomatrix(img, distances=[5], angles=[0], levels=levels, symmetric=True, normed=True)

    # Compute texture features
    texture_features = {
        "contrast": graycoprops(glcm, 'contrast')[0, 0],
        "correlation": graycoprops(glcm, 'correlation')[0, 0],
        "energy": graycoprops(glcm, 'energy')[0, 0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0, 0]
    }
    return texture_features

def visualize_sample_UMAP(embeddings, labels, save_path):
    reducer = umap.UMAP(n_jobs=1,  # Use single thread
                        random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], \
                    hue=labels, palette=['green', 'red'], \
                    s=25, alpha=0.95, edgecolor='k',linewidth=0.3)
    plt.title('UMAP projection of the embeddings', fontsize=15)
    plt.savefig(save_path)
    plt.close()

def visualize_train_test_sample_UMAP(train_embeddings, train_labels, test_embeddings, test_labels, save_path):
    reducer = umap.UMAP(n_jobs=4)
    embedding_2d = reducer.fit_transform(train_embeddings)
    embedding_2d_test = reducer.transform(test_embeddings)

    plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], \
                    hue=train_labels, palette=['green', 'red'], \
                    s=25, alpha=0.95, edgecolor='k',linewidth=0.3)
    plt.title('UMAP projection of the embeddings', fontsize=15)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embedding_2d_test[:, 0], y=embedding_2d_test[:, 1], \
                    hue=test_labels, palette=['green', 'red'], \
                    s=25, alpha=0.95, edgecolor='k',linewidth=0.3)
    plt.title('UMAP projection of the embeddings', fontsize=15)
    plt.savefig(save_path)
    plt.close()

def visualize_train_test_sample_tSNE(train_embeddings, train_labels, test_embeddings, test_labels, save_path):
    tsne = openTSNE.TSNE(n_components=2, perplexity=25, random_state=42, n_jobs=4)
    embedding_2d = tsne.fit(train_embeddings)
    embedding_2d_test = embedding_2d.transform(test_embeddings)

    plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1],
                    hue=train_labels, palette=['green', 'red'], 
                    s=25, alpha=0.95, edgecolor='k',linewidth=0.3)
    plt.title('t-SNE projection of the embeddings', fontsize=15)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embedding_2d_test[:, 0], y=embedding_2d_test[:, 1], \
                    hue=test_labels, palette=['green', 'red'], \
                    s=25, alpha=0.95, edgecolor='k',linewidth=0.3)
    plt.title('t-SNE projection of the embeddings', fontsize=15)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':    
    from config import DATA_ROOT_DIR
    data_dir = os.path.join(DATA_ROOT_DIR, "original_data/data_pixels")
    output_dir = os.path.join(DATA_ROOT_DIR, "output")

    # List all CSV files in the data directory
    # csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    csv_files = sorted(csv_files)
    for csv_file in csv_files:
        # csv_file = "502861new.txt"
        fn = os.path.join(data_dir, csv_file)
        # Read the CSV file
        # df = pd.read_csv(fn, sep=',')
        df = pd.read_csv(fn, sep='\t')
        # Please note: Set center to True to center the image, 
        # otherwise the embedding can be overwhemled by the location of the lesion.
        images, delta_times = reconstruct_images_from_csv(df, center=True)
        
        # Save the reconstructed images to separate files
        save_reconstructed_images(images, csv_file, output_dir)

        # Sort images by delta_time
        sorted_images = dict(sorted(images.items(), key=lambda item: float(item[0].split('_')[1])))
    
        # Create subplots
        num_phases = len(sorted_images)
        fig, axes = plt.subplots(1, num_phases, figsize=(15, 5), dpi=250)
        
        if num_phases == 1:
            axes = [axes]
        
        for ax, (phase_delta, image) in zip(axes, sorted_images.items()):
            phase, delta_time = phase_delta.split('_')
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Phase: {phase}\nDelta Time: {delta_time} s')
            ax.axis('off')
        
        plt.tight_layout()
        # plt.show()
        
        # Save the figure to the output directory
        output_fn = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
        plt.savefig(output_fn)
        plt.close()

    print("Done!")

    