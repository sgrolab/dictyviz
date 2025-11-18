# Dicty data functions for ome-zarr datasets
import copy
import os
import traceback

import xml.etree.ElementTree as et
import cv2
import cmapy
from PIL import ImageFont, ImageDraw, Image
from dask.distributed import Client, wait
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from .utils import get_cropping_dims, get_voxel_dims, get_slice_depth, get_imaging_freq


class Channel:
    def __init__(self, name, n_channel, scale_max, scale_min, gamma=1.0, invert_channel=False, voxel_dims=None):
        self.name = name
        self.n_channel = n_channel
        self.scale_max = scale_max
        self.scale_min = scale_min
        self.gamma = gamma
        self.invert_channel = invert_channel
        self.voxel_dims = voxel_dims


def calc_max_projections(root, max_projections_root, res_lvl=0):

    # define resolution level
    res_array = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine res_array
    crop_dims = get_cropping_dims(max_projections_root.store.path + '/../../parameters.json')
    if crop_dims:
        res_array = res_array[:, :, crop_dims[2][0]:crop_dims[2][1], crop_dims[1][0]:crop_dims[1][1], crop_dims[0][0]:crop_dims[0][1]]

    # get dataset dimensions
    len_t, len_ch, len_z, len_y, len_x = res_array.shape

    # create zarr arrays for each max projection 
    max_z = max_projections_root.zeros('maxz',shape=(len_t,len_ch,2,len_y,len_x),chunks=(1,len_ch,2,len_y,len_x))
    max_x = max_projections_root.zeros('maxx',shape=(len_t,len_ch,len_z,len_y),chunks=(1,len_ch,len_z,len_y))
    max_y = max_projections_root.zeros('maxy',shape=(len_t,len_ch,len_z,len_x),chunks=(1,len_ch,len_z,len_x))

    # iterate through each timepoint and compute max projections
    for i in tqdm(range(len_t)):
        for j in range(len_ch):
            frame = res_array[i, j, :, :, :]
            max_z[i,j] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
            max_x[i,j] = np.max(frame,axis=2)
            max_y[i,j] = np.max(frame,axis=1)  

def calc_optical_flow_max_projections(max_projections_root, optical_folder, crop_id=''):
    """Create max projections specifically from optical flow RGBM (rgb and magnitude) arrays"""
    
    print(f"🔍 Looking for optical flow data in: {optical_folder}")
    
    # Find all frame directories
    frame_dirs = []
    if os.path.exists(optical_folder):
        for item in os.listdir(optical_folder):
            item_path = os.path.join(optical_folder, item)
            if os.path.isdir(item_path) and item.isdigit():
                frame_dirs.append(int(item))
    
    if not frame_dirs:
        print(f"No frame directories found in {optical_folder}")
        return
    
    frame_dirs.sort()
    print(f"📊 Found {len(frame_dirs)} frames: {frame_dirs[:5]}{'...' if len(frame_dirs) > 5 else ''}")
    
    # Get dimensions from first available RGBM file
    sample_frame = frame_dirs[0]
    sample_rgbm_file = os.path.join(optical_folder, str(sample_frame), "optical_flow_rgbm.npy")
    
    if not os.path.exists(sample_rgbm_file):
        print(f"Sample RGBM file not found: {sample_rgbm_file}")
        return
    
    # Load sample to get dimensions
    sample_rgbm = np.load(sample_rgbm_file)
    len_z, len_y, len_x, num_channels = sample_rgbm.shape
    len_t = len(frame_dirs)
    
    print(f"📐 Dataset dimensions:")
    print(f"   Frames (T): {len_t}")
    print(f"   Z-slices: {len_z}")
    print(f"   Y: {len_y}")
    print(f"   X: {len_x}")
    print(f"   Channels: {num_channels} (RGBM)")
    
    # Check for cropping parameters
    parent_dir = os.path.dirname(optical_folder)
    crop_dims = get_cropping_dims(os.path.join(parent_dir, 'parameters.json'))
    if crop_dims:
        print(f"🔧 Applying cropping: {crop_dims}")
        len_z = crop_dims[2][1] - crop_dims[2][0]
        len_y = crop_dims[1][1] - crop_dims[1][0]
        len_x = crop_dims[0][1] - crop_dims[0][0]
        print(f"📐 Cropped dimensions: Z={len_z}, Y={len_y}, X={len_x}")
    
    # Create zarr arrays for optical flow RGBM max projections
    print("📝 Creating zarr arrays for max projections...")
    
    flow_max_z = max_projections_root.zeros('flow_maxz', shape=(len_t, 4, len_y, len_x), 
                                       chunks=(1, 4, len_y, len_x))
    flow_max_x = max_projections_root.zeros('flow_maxx', shape=(len_t, 4, len_z, len_y), 
                                       chunks=(1, 4, len_z, len_y))
    flow_max_y = max_projections_root.zeros('flow_maxy', shape=(len_t, 4, len_z, len_x), 
                                       chunks=(1, 4, len_z, len_x))
    
    print("Zarr arrays created successfully")
    
    # Iterate through each timepoint and compute max projections
    print("🎨 Computing max projections...")
    
    for i, frame_num in enumerate(tqdm(frame_dirs, desc="Processing frames")):
        frame_dir = os.path.join(optical_folder, str(frame_num))
        rgbm_file = os.path.join(frame_dir, "optical_flow_rgbm.npy")
        
        if os.path.exists(rgbm_file):
            try:
                # Load RGBM data: (Z, Y, X, 4) where 4 = [Red, Green, Blue, Magnitude]
                rgbm_data = np.load(rgbm_file)
                
                # Apply cropping if specified
                if crop_dims:
                    rgbm_data = rgbm_data[crop_dims[2][0]:crop_dims[2][1], 
                                         crop_dims[1][0]:crop_dims[1][1], 
                                         crop_dims[0][0]:crop_dims[0][1], :]
                
                # Compute max projections for each channel (R, G, B, M)
                for j in range(4):
                    channel_data = rgbm_data[:, :, :, j]  # (Z, Y, X)
                    
                    # Max projections along each axis
                    flow_max_z[i, j] = np.max(channel_data, axis=0)  # (Y, X) - along Z
                    flow_max_x[i, j] = np.max(channel_data, axis=2)  # (Z, Y) - along X
                    flow_max_y[i, j] = np.max(channel_data, axis=1)  # (Z, X) - along Y
                    
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                # Fill with zeros for this frame
                flow_max_z[i, :] = 0
                flow_max_x[i, :] = 0
                flow_max_y[i, :] = 0
        else:
            print(f"RGBM file not found for frame {frame_num}")
            # Fill with zeros for this frame
            flow_max_z[i, :] = 0
            flow_max_x[i, :] = 0
            flow_max_y[i, :] = 0
    
    print("Optical flow max projections completed!")
    print(f"Created arrays:")
    print(f"   flow_maxz: {flow_max_z.shape}")
    print(f"   flow_maxx: {flow_max_x.shape}")
    print(f"   flow_maxy: {flow_max_y.shape}")

def calc_maxs(res_array, i, j, max_z, max_x, max_y):
    frame = res_array[i, j, :, :, :]
    max_z[i,j] = [np.max(frame,axis=0), np.argmax(frame,axis=0)]
    max_x[i,j] = np.max(frame,axis=2)
    max_y[i,j] = np.max(frame,axis=1)    

def calc_max_projections_dask(root, max_projections_root, client, res_lvl=0):

    # define resolution level
    res_array = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine res_array
    crop_dims = get_cropping_dims(max_projections_root.store.path + '/../../parameters.json')
    if crop_dims:
        res_array = res_array[:, :, crop_dims[2][0]:crop_dims[2][1], crop_dims[1][0]:crop_dims[1][1], crop_dims[0][0]:crop_dims[0][1]]

    # get dataset dimensions
    len_t, len_ch, len_z, len_y, len_x = res_array.shape

    # create zarr arrays for each max projection 
    max_z = max_projections_root.zeros('maxz',shape=(len_t,len_ch,2,len_y,len_x),chunks=(1,len_ch,2,len_y,len_x))
    max_x = max_projections_root.zeros('maxx',shape=(len_t,len_ch,len_z,len_y),chunks=(1,len_ch,len_z,len_y))
    max_y = max_projections_root.zeros('maxy',shape=(len_t,len_ch,len_z,len_x),chunks=(1,len_ch,len_z,len_x))

    # iterate through each timepoint and compute max projections
    futures = []
    for i in tqdm(range(len_t)):
        for j in range(len_ch):
            futures.append(client.submit(calc_maxs, res_array, i, j, max_z, max_x, max_y))
    wait(futures)    


def calc_sliced_max_projections(root, sliced_max_projections_root, res_lvl=0):

    # define resolution level
    res_array = root['0'][str(res_lvl)]

    #if cropping, get crop parameters from parameters.json and redefine res_array
    crop_dims = get_cropping_dims(sliced_max_projections_root.store.path + '/../../parameters.json')
    if crop_dims:
        res_array = res_array[:, :, crop_dims[2][0]:crop_dims[2][1], crop_dims[1][0]:crop_dims[1][1], crop_dims[0][0]:crop_dims[0][1]]

    # get dataset dimensions
    len_t, len_ch, len_z, len_y, len_x = res_array.shape

    # get slice depth from parameters.json
    slice_depth = get_slice_depth(sliced_max_projections_root.store.path + '/../../parameters.json')
    voxel_dims = get_voxel_dims(root.store.path + '/OME/METADATA.ome.xml')

    # set number of slices
    if slice_depth == "auto":
        n_slices_x = 20
        n_slices_y = 20
        slice_depth_x = len_x//(n_slices_x-1)
        slice_depth_y = len_y//(n_slices_y-1)
    else:
        slice_depth_x = int(slice_depth//voxel_dims[0])
        n_slices_x = len_x//slice_depth_x
        slice_depth_y = int(slice_depth//voxel_dims[1])
        n_slices_y = len_y//slice_depth_y

    # create zarr arrays for each max projection
    sliced_max_x = sliced_max_projections_root.zeros('sliced_maxx',shape=(len_t,len_ch,n_slices_x,len_z,len_y),chunks=(1,1,2,len_z,len_y))
    sliced_max_y = sliced_max_projections_root.zeros('sliced_maxy',shape=(len_t,len_ch,n_slices_y,len_z,len_x),chunks=(1,1,2,len_z,len_x))

    for i in tqdm(range(len_t)):
        for j in range(len_ch):
            for k in range(n_slices_x-1):
                range_x = [k*slice_depth_x, (k+1)*slice_depth_x]
                frame = res_array[i,j,:,:,range_x[0]:range_x[1]]
                sliced_max_x[i,j,k] = np.max(frame,axis=2)
            #fill last chunk with the rest of the data
            range_x = [(n_slices_x-1)*slice_depth_x, len_x]
            frame = res_array[i,j,:,:,range_x[0]:range_x[1]]
            sliced_max_x[i,j,n_slices_x-1] = np.max(frame,axis=2)

            for k in range(n_slices_y-1):
                range_y = [k*slice_depth_y, (k+1)*slice_depth_y]
                frame = res_array[i,j,:,range_y[0]:range_y[1],:]
                sliced_max_y[i,j,k] = np.max(frame,axis=1)
            #fill last chunk with the rest of the data
            range_y = [(n_slices_y-1)*slice_depth_y, len_y]
            frame = res_array[i,j,:,range_y[0]:range_y[1],:]
            sliced_max_y[i,j,n_slices_y-1] = np.max(frame,axis=1)

def generate_unique_filename(filename, ext):
    i = 1
    while os.path.exists(filename + ext):
        if i == 1:
            filename = filename +'_1'
        else:
            filename = filename[:-1] + str(i)
        i += 1
    return filename + ext

def calc_auto_contrast(max_proj, n_channel):
    first_frame = max_proj[0, n_channel, 0]

    # define histogram of pixel values in first frame
    hist_min = np.min(first_frame)
    hist_max = np.max(first_frame)
    bin_size = (hist_max - hist_min) / 256
    histogram = np.histogram(first_frame, bins = 256, range=(hist_min, hist_max))[0]

    # define limit and threshold
    height, width = first_frame.shape[:2]
    pixel_count = height * width
    limit = pixel_count/10
    threshold = int(pixel_count/5000)

    # find the bin for the min and max contrast values
    bin = -1
    found_min_bin = False
    while not found_min_bin and bin < 255:
        bin += 1
        count_in_bin = histogram[bin]
        if count_in_bin > limit:
            count_in_bin = 0
        found_min_bin = count_in_bin > threshold
    h_min_bin = bin
    
    bin = 256
    found_max_bin = False
    while not found_max_bin and bin > 0:
        bin -= 1
        count_in_bin = histogram[bin]
        if count_in_bin > limit:
            count_in_bin = 0
        found_max_bin = count_in_bin > threshold
    h_max_bin = bin

    # find scale_min and scale_max based on h_min_bin and h_max_bin
    if h_max_bin > h_min_bin:
        scale_min = hist_min + (h_min_bin * bin_size)
        scale_max = hist_min + (h_max_bin * bin_size)
    # bad cases: h_max_bin is same or less than h_min_bin, just use the min and max of the histogram
    else:
        scale_min = hist_min
        scale_max = hist_max

    print('scale_min:', scale_min)
    print('scale_max:', scale_max)

    return scale_min, scale_max
    

def get_projection_dimensions(root):
    # return the dimensions of the max projections
    if 'maxx' in root:
        max_x = root['maxx']
        max_y = root['maxy']
    else:
        max_x = root['sliced_maxx']
        max_y = root['sliced_maxy']
    len_t = max_x.shape[0]
    len_z = max_x.shape[-2]
    len_y = max_x.shape[-1]
    len_x = max_y.shape[-1]
    return len_t, len_z, len_y, len_x

def adjust_contrast(im, scale_max, scale_min, gamma):
    im = np.clip(im,scale_min,scale_max)
    back_sub = im - scale_min
    back_sub[np.where(back_sub<0)] = 0
    scaled_im = np.divide(back_sub,scale_max-scale_min)
    gamma_corrected_im = np.power(scaled_im,gamma)
    contrasted_im = np.multiply(gamma_corrected_im,255).astype('uint8')
    return(contrasted_im)


class ScaleBar:
    def __init__(self, pos_y, pos_x, length, px_per_micron, font):
        self.pos_y = pos_y
        self.pos_x = pos_x
        self.length = length
        self.px_per_micron = px_per_micron
        self.font = font
        self.length_in_px = round(length*px_per_micron)
        self.height_in_px = int((length*px_per_micron)//10)
        self.units = '\u03BCm'
        self.text = str(self.length) + self.units

    def _set_font(self):
        #TODO: add a style sheet and get font from there
        module_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(module_dir, '..', 'fonts', 'Lato2OFL', 'Lato-Black.ttf')
        font_size = 1
        self.font = ImageFont.truetype(font_path, font_size)
        while self.font.getlength(self.text) <= self.length_in_px/2:
            font_size += 1
            self.font = ImageFont.truetype(font_path, font_size)
        return

    def _add_scale_bar(self, frame):
        # add bar
        frame[self.pos_y-self.height_in_px:self.pos_y, self.pos_x-self.length_in_px:self.pos_x,:] = 255
        # convert length to mm if larger than 1000
        if self.length >= 1000:
            self.text = str(int(self.length/1000)) + 'mm'
        # add text
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text_width = draw.textlength(self.text, font=self.font)
        text_height, _ = self.font.getmetrics()
        text_pos = (self.pos_x-(self.length_in_px//2)-(text_width//2), self.pos_y-round(self.height_in_px*1.3)-text_height)
        draw.text(text_pos, self.text, font=self.font, fill=(255, 255, 255))
        frame = np.array(img_pil)
        return frame
      
def get_scale_bar_length(root, voxel_dims):
    #TODO: add scaling factor for sliced movies where the scale bar should be smaller
    #approx_scale_bar_length = proj_dims_um[1]/scale_factor
    #alternatively, for sliced movies, proj_dims should be switched out with movie_dims
    scale_bar_lengths = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000] # in um

    proj_dims_px = get_projection_dimensions(root)
    proj_dims_um = [proj_dims_px[3]*voxel_dims[0], proj_dims_px[2]*voxel_dims[1], proj_dims_px[1]*voxel_dims[2]]
    approx_scale_bar_length = proj_dims_um[1]/5
    scale_bar_length = min(scale_bar_lengths, key=lambda x:abs(x-approx_scale_bar_length))
    return scale_bar_length


def calculate_time_stamp(frame_index, imaging_freq, total_frames):
    """
    Calculate timestamp string from frame index and imaging frequency.
    
    Args:
        frame_index (int): Frame number (0-based)
        imaging_freq (float): Time between frames in minutes
        
    Returns:
        str: Formatted timestamp string
    """
    # Calculate maximum timestamp to determine format
    max_total_minutes = (total_frames - 1) * imaging_freq
    max_hours = int(max_total_minutes // 60)
    max_remaining_minutes = max_total_minutes % 60
    max_minutes = int(max_remaining_minutes)
    max_seconds = int((max_remaining_minutes - max_minutes) * 60)

    show_hours = max_hours > 0
    show_seconds = max_seconds > 0

    # Calculate current timestamp
    total_minutes = frame_index * imaging_freq
    hours = int(total_minutes // 60)
    remaining_minutes = total_minutes % 60
    minutes = int(remaining_minutes)
    seconds = int((remaining_minutes - minutes) * 60)
    
    # Format based on maximum needed format
    if show_hours:
        if show_seconds:
            return f'{hours:02d}h:{minutes:02d}m:{seconds:02d}s'
        else:
            return f'{hours:02d}h:{minutes:02d}m'
    else:
        if show_seconds:
            return f'{minutes:02d}m:{seconds:02d}s'
        else:
            return f'{minutes:02d}m'


def add_time_stamp(frame, time_stamp_pos, t, font):
    # time stamp
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text(time_stamp_pos, t, font=font, fill=(255, 255, 255))
    frame = np.array(img_pil)
    return frame

def scale_xzyz(im, z_to_xy_ratio):
    # scale the XZ and YZ projections to match the XY projection
    scaled_im = np.zeros([int(round(im.shape[0]*z_to_xy_ratio)), im.shape[1]])
    for i in range(im.shape[0]):
        scaled_im[int(round(i*z_to_xy_ratio)):int(round((i+1)*z_to_xy_ratio)),:] = im[i,:]
    return scaled_im


def make_ortho_max_video_clean(root, channel, cmap, ext='.avi'):

    filename = generate_unique_filename(channel.name + '_orthomax_clean_' + cmap, ext)
    n_channel = channel.n_channel
    voxel_dims = channel.voxel_dims
    scale_max = channel.scale_max
    scale_min = channel.scale_min
    gamma = channel.gamma

    imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')

    max_z = root['maxz']
    
    len_t, len_z, len_y, len_x = get_projection_dimensions(root)

    movie_width = len_x 
    movie_height = len_y

    # calc scale_min, scale_max, and gamma if not provided
    if scale_min == "None" or scale_max == "None":
        scale_min, scale_max = calc_auto_contrast(max_z, n_channel)
    if gamma == "None":
        gamma = 1
    
    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movie_width,movie_height),1)

    try: 
        for i in tqdm(range(len_t)):

            # initialize frame 
            im = np.zeros([movie_height,movie_width])

            # copy max projections 
            im[:movie_height,0:len_x] = copy.copy(max_z[i,n_channel,0])
            
            contrasted_im = adjust_contrast(im, scale_max, scale_min, gamma)

            # invert if rock channel
            if channel.name == 'rocks':
                contrasted_im = 255 - contrasted_im

            frame = cv2.applyColorMap(contrasted_im,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def make_ortho_max_video(root, channel, cmap, ext='.avi'):

    filename = generate_unique_filename(channel.name + '_orthomax_' + cmap, ext)
    n_channel = channel.n_channel
    voxel_dims = channel.voxel_dims
    scale_max = channel.scale_max
    scale_min = channel.scale_min
    gamma = channel.gamma

    imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    max_z = root['maxz']
    max_y = root['maxy']
    max_x = root['maxx']
    
    len_t, len_z, len_y, len_x = get_projection_dimensions(root)

    # calc scaled Z dimension
    z_to_xy_ratio = voxel_dims[2]/voxel_dims[0]
    scaled_len_z = int(round(len_z*z_to_xy_ratio))
    
    gap = 20

    movie_width = len_x + scaled_len_z + gap
    movie_height = len_y + scaled_len_z + gap

    # calc scale_min, scale_max, and gamma if not provided
    if scale_min == "None" or scale_max == "None":
        scale_min, scale_max = calc_auto_contrast(max_z, n_channel)
    if gamma == "None":
        gamma = 1

    # define scale bars
    scale_bar_length = get_scale_bar_length(root, channel.voxel_dims)
    scale_bar_xy = ScaleBar(
        pos_y = movie_height,
        pos_x = len_x,
        length = scale_bar_length,
        px_per_micron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xy._set_font()
    scale_bar_xz = ScaleBar(
        pos_y = scaled_len_z,
        pos_x = len_x + gap + scaled_len_z,
        length = int(len_z*channel.voxel_dims[2]),
        px_per_micron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xz._set_font()
    
    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movie_width,movie_height),1)

    try: 
        for i in tqdm(range(len_t)):

            # initialize frame 
            im = np.zeros([movie_height,movie_width])

            # copy max projections 
            im_xz = copy.copy(np.flip(max_y[i,n_channel],axis=0))
            im[0:scaled_len_z,0:len_x] = scale_xzyz(im_xz, z_to_xy_ratio)
            im[(scaled_len_z+gap):movie_height,0:len_x] = copy.copy(max_z[i,n_channel,0])
            im_yz = copy.copy(max_x[i,n_channel])
            im[(scaled_len_z+gap):movie_height,(len_x+gap):movie_width] = np.transpose(scale_xzyz(im_yz, z_to_xy_ratio))
            
            contrasted_im = adjust_contrast(im, scale_max, scale_min, gamma)

            # invert if rock channel
            if channel.invert_channel:
                contrasted_im = 255 - contrasted_im

            frame = cv2.applyColorMap(contrasted_im,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # add scale bars
            frame = scale_bar_xy._add_scale_bar(frame)
            frame = scale_bar_xz._add_scale_bar(frame)

            # time stamp
            t = calculate_time_stamp(i, imaging_freq, len_t)
            time_stamp_pos = (0, scaled_len_z+gap)
            frame = add_time_stamp(frame, time_stamp_pos, t, scale_bar_xy.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def process_single_slice(slice_data, tiles_y, tiles_x):
    """Process a single 2D slice by normalizing each tile separately."""
    height, width = slice_data.shape
    normalized_slice = np.zeros_like(slice_data)
    
    tile_height = height // tiles_y
    tile_width = width // tiles_x
    
    # Process each tile independently
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Get tile boundaries
            y_start = i * tile_height
            y_end = min((i + 1) * tile_height, height)
            x_start = j * tile_width
            x_end = min((j + 1) * tile_width, width)
            
            # Extract and normalize this tile
            tile = slice_data[y_start:y_end, x_start:x_end]
            tile_mean = np.mean(tile)
            
            if np.abs(tile_mean) > 1e-6:  # Only normalize if tile isn't uniform
                normalized_tile = tile - tile_mean
                
                # Enhance contrast using percentile clipping
                p_low, p_high = np.percentile(normalized_tile, [5, 95])
                if p_high > p_low:
                    normalized_tile = np.clip(normalized_tile, p_low, p_high)
                    normalized_tile = (normalized_tile - p_low) / (p_high - p_low)
                else:
                    normalized_tile = np.zeros_like(tile)
            else:
                normalized_tile = np.zeros_like(tile)
            
            # Store the normalized tile back
            normalized_slice[y_start:y_end, x_start:x_end] = normalized_tile
    
    return normalized_slice

def normalize_vz_with_tiling(vz_data, tiles_y=4, tiles_x=3, sigma_smooth=3, sigma_blend=1):
    """
    Normalize z-flow data using per-tile mean subtraction to eliminate tiling artifacts.
    
    This function helps remove systematic errors that appear as grid patterns
    in the z-component of optical flow data.
    """
    # Smooth the data first to reduce noise
    vz_smoothed = gaussian_filter(vz_data, sigma=sigma_smooth)
    
    # Handle both 2D and 3D data
    if len(vz_data.shape) == 3:
        depth, height, width = vz_smoothed.shape
        vz_normalized = np.zeros_like(vz_smoothed)
        
        for d in range(depth):
            vz_normalized[d] = process_single_slice(vz_smoothed[d], tiles_y, tiles_x)
    else:
        vz_normalized = process_single_slice(vz_smoothed, tiles_y, tiles_x)
    
    # Smooth tile boundaries to blend them together
    vz_normalized = gaussian_filter(vz_normalized, sigma=sigma_blend)
    return vz_normalized


def enhance_channel_contrast(channel):
    """Enhance the contrast of a single channel using percentile normalization."""
    p_low, p_high = np.percentile(channel, [5, 95])
    if p_high > p_low:
        enhanced = np.clip(channel, p_low, p_high)
        enhanced = (enhanced - p_low) / (p_high - p_low)
        enhanced = enhanced ** 1.6  # Slight gamma correction to reduce brightness
        return enhanced
    return channel

def balance_rgb_channels(red, green, blue):
    """Balance RGB channels to prevent any single color from dominating."""
    # Calculate mean intensity for each channel
    red_mean = np.mean(red)
    green_mean = np.mean(green) 
    blue_mean = np.mean(blue)
    max_mean = max(red_mean, green_mean, blue_mean)
    
    if max_mean > 1e-6:
        # Scale channels relative to the brightest one
        if red_mean > 1e-6:
            red = red * (max_mean / red_mean) * 0.9    # Reduce red slightly
        if green_mean > 1e-6:
            green = green * (max_mean / green_mean) * 1.1  # Boost green slightly
        if blue_mean > 1e-6:
            blue = blue * (max_mean / blue_mean) * 0.6     # Reduce blue significantly
    
    return red, green, blue

def smooth_image(image, sigma=2.5, preserve_edges=True, passes=3):
    """
    Apply smoothing to improve visualization quality without resizing.
    Uses Gaussian filtering with optional edge preservation. Iterates over passes to improve smoothing 
    
    Args:
        image: Input image (grayscale or color)
        sigma: Smoothing strength (higher = more smoothing)
        preserve_edges: If True, uses edge-preserving filtering
    
    Returns:
        Smoothed image of same dimensions as input
    """
    # Handle different image types
    smoothed = image.copy()
    
    for _ in range(passes):
        if len(smoothed.shape) == 3:  # Color image
            if preserve_edges and cv2.__version__ >= '3.0.0':
                smoothed = cv2.bilateralFilter(smoothed.astype(np.float32), 
                                              d=0, 
                                              sigmaColor=sigma*15, 
                                              sigmaSpace=sigma)
        else:
            # Apply Gaussian smoothing to each channel
            smoothed = np.zeros_like(image, dtype=np.float32)
            for c in range(3):
                smoothed[:,:,c] = gaussian_filter(image[:,:,c], sigma=sigma)
    else:  # Grayscale
        if preserve_edges and cv2.__version__ >= '3.0.0':
            # Edge-preserving filter for grayscale images
            smoothed = cv2.bilateralFilter(image.astype(np.float32), 
                                           d=0,  # Automatic diameter
                                           sigmaColor=sigma*15, 
                                           sigmaSpace=sigma)
        else:
            # Standard Gaussian smoothing
            smoothed = gaussian_filter(image, sigma=sigma)
    
    # Ensure output is in same format as input
    if image.dtype == np.uint8:
        smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
    
    return smoothed

def make_ortho_max_optical_flow_video(root, channel, ext='.mp4'):
    """
    Create a video showing optical flow in three orthogonal projections.
    
    Layout:
    - Top: YZ projection (side view)
    - Bottom-left: XY projection (top-down view)  
    - Bottom-right: XZ projection (front view)
    """
    # Generate unique filename
    filename = generate_unique_filename(channel.name + '_optical_flow_orthomax_', ext)
    voxel_dims = channel.voxel_dims

    # Get imaging frequency for timestamps
    try:
        imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')
    except:
        imaging_freq = 10

    # Load the optical flow data
    flow_max_z = root['flow_maxz']  # (T,4,Y,X) - XY projections over time
    flow_max_y = root['flow_maxy']  # (T,4,Z,X) - XZ projections over time  
    flow_max_x = root['flow_maxx']  # (T,4,Z,Y) - YZ projections over time
    
    print(f"Flow data dimensions:")
    print(f"  XY projection: {flow_max_z.shape}")
    print(f"  XZ projection: {flow_max_y.shape}")
    print(f"  YZ projection: {flow_max_x.shape}")
    
    # Get dimensions
    len_t = flow_max_z.shape[0]
    len_z = flow_max_x.shape[2]
    len_y = flow_max_z.shape[2] 
    len_x = flow_max_z.shape[3]

    # Calculate total video dimensions
    gap = 20  # Space between projections
    movie_width = len_x + gap + len_z
    movie_height = len_z + gap + len_y
    
    print(f"Video layout: {movie_width}x{movie_height}")

    # Setup scale bar
    scale_bar_lengths = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000]
    proj_dims_um = [len_x*voxel_dims[0], len_y*voxel_dims[1], len_z*voxel_dims[2]]
    approx_scale_bar_length = proj_dims_um[1]/5
    scale_bar_length = min(scale_bar_lengths, key=lambda x:abs(x-approx_scale_bar_length))

    scale_bar_xy = ScaleBar(
        pos_y = movie_height - 10,
        pos_x = len_x - 10, 
        length = scale_bar_length,
        px_per_micron = 1/voxel_dims[0],
        font = None,
    )
    scale_bar_xy._set_font()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext.lower() == '.mp4' else cv2.VideoWriter_fourcc(*'MJPG')
    vid = cv2.VideoWriter(filename, fourcc, 10, (movie_width, movie_height), 1)
    
    try: 
        for i in tqdm(range(len_t), desc="Creating video frames"):
            # Initialize empty frame
            frame = np.zeros([movie_height, movie_width, 3], dtype=np.uint8)

            # Extract RGB channels for each projection at this time point
            red_xy, green_xy, blue_xy = flow_max_z[i, 0], flow_max_z[i, 1], flow_max_z[i, 2]
            red_xz, green_xz, blue_xz = flow_max_y[i, 0], flow_max_y[i, 1], flow_max_y[i, 2] 
            red_yz, green_yz, blue_yz = flow_max_x[i, 0], flow_max_x[i, 1], flow_max_x[i, 2]
            
            # Process XY projection (standard contrast enhancement)
            red_xy = enhance_channel_contrast(red_xy)
            green_xy = enhance_channel_contrast(green_xy)
            blue_xy = enhance_channel_contrast(blue_xy)
            
            # Process XZ projection (with z-component normalization for blue channel)
            red_xz = enhance_channel_contrast(red_xz)
            green_xz = enhance_channel_contrast(green_xz)
            blue_xz = normalize_vz_with_tiling(blue_xz)  # Special processing for z-flow
            red_xz, green_xz, blue_xz = balance_rgb_channels(red_xz, green_xz, blue_xz)
            blue_xz = blue_xz * 0.7  # Further reduce blue dominance
            
            # Process YZ projection (with z-component normalization for blue channel)
            red_yz = enhance_channel_contrast(red_yz)
            green_yz = enhance_channel_contrast(green_yz)
            blue_yz = normalize_vz_with_tiling(blue_yz)  # Special processing for z-flow
            red_yz, green_yz, blue_yz = balance_rgb_channels(red_yz, green_yz, blue_yz)
            blue_yz = blue_yz * 0.7  # Further reduce blue dominance
            
            # Convert to 8-bit values
            brightness = 1.0
            def to_uint8(channel):
                return np.clip(channel * 255 * brightness, 0, 255).astype(np.uint8)
            
            # Convert all channels
            red_xy, green_xy, blue_xy = to_uint8(red_xy), to_uint8(green_xy), to_uint8(blue_xy)
            red_xz, green_xz, blue_xz = to_uint8(red_xz), to_uint8(green_xz), to_uint8(blue_xz)
            red_yz, green_yz, blue_yz = to_uint8(red_yz), to_uint8(green_yz), to_uint8(blue_yz)

            # Create RGB images (BGR format for OpenCV)
            rgb_xy = np.stack([blue_xy, green_xy, red_xy], axis=-1)
            rgb_xz = np.stack([blue_xz, green_xz, red_xz], axis=-1)
            rgb_yz = np.stack([blue_yz, green_yz, red_yz], axis=-1)

            # Place XY projection (bottom-left, main view)
            try:
                smoothed_xy = smooth_image(rgb_xy)
                y_start = len_z + gap
                frame[y_start:y_start + len_y, 0:len_x] = smoothed_xy
            except Exception as e:
                print(f"Error placing XY projection: {e}")

            # Place XZ projection (bottom-right strip)
            try:
                # Flip vertically so Z increases upward
                rgb_xz_flipped = np.flip(rgb_xz, axis=0)
                smoothed_xz = smooth_image(rgb_xz_flipped)
                frame[0:len_z, 0:len_x] = smoothed_xz
            except Exception as e:
                print(f"Error placing XZ projection: {e}")

            # Place YZ projection (top strip)
            try:
                # Transpose from (Z,Y,3) to (Y,Z,3) for proper orientation
                rgb_yz_transposed = np.transpose(rgb_yz, (1, 0, 2))
                smoothed_yz = smooth_image(rgb_yz_transposed)
                y_start = len_z + gap
                x_start = len_x + gap
                frame[y_start:y_start + len_y, x_start:x_start + len_z] = smoothed_yz
            except Exception as e:
                print(f"Error placing YZ projection: {e}")

            # Add scale bar and timestamp
            frame = scale_bar_xy._add_scale_bar(frame)
            t = calculate_time_stamp(i, imaging_freq, len_t)
            frame = add_time_stamp(frame, (10, 20), t, scale_bar_xy.font)

            # Write frame to video
            vid.write(frame)

        # Cleanup
        vid.release()
        cv2.destroyAllWindows()
        print(f"✓ Video saved: {filename}")

    except Exception as e:
        print(f"Error creating video: {e}")
        traceback.print_exc()
        try:
            vid.release()
        except:
            pass
        cv2.destroyAllWindows()

def make_sliced_ortho_max_videos(root, channel, dim, cmap, ext='.avi'):

    filename = generate_unique_filename(channel.name + "_" + dim + '_sliced_orthomax_' + cmap, ext)
    n_channel = channel.n_channel
    scale_max = channel.scale_max
    scale_min = channel.scale_min
    gamma = channel.gamma

    imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')
    
    sliced_max = root['sliced_max'+dim]

    len_t, len_z, _, _ = get_projection_dimensions(root)

    gap = 20

    # calc scale_min, scale_max, and gamma if not provided
    if scale_min == "None" or scale_max == "None":
        scale_min, scale_max = calc_auto_contrast(sliced_max, n_channel)
    if gamma == "None":
        gamma = 1 
        
    n_slices = sliced_max.shape[2]

    movie_width = sliced_max.shape[-1]
    movie_height = (len_z * n_slices) + (gap * (n_slices-1))

    # define scale bars
    scale_bar_length = get_scale_bar_length(root, channel.voxel_dims)
    scale_bar_xy = ScaleBar(
        posY = movie_height,
        posX = movie_width,
        length = scale_bar_length,
        pxPerMicron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xy._set_font()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movie_width,movie_height),1)

    try:
        for i in tqdm(range(len_t)):

            # initialize frame
            im = np.zeros([movie_height,movie_width])

            # copy max projections 
            for j in range(n_slices):
                im[(len_z*j+gap*j):(len_z*(j+1)+gap*j),:] = copy.copy(np.flip(sliced_max[i,n_channel,j], axis=0))

            # adjust contrast
            contrasted_im = adjust_contrast(im, scale_max, scale_min, gamma)

            # invert if rock channel
            if channel.invert_channel:
                contrasted_im = 255 - contrasted_im

            frame = cv2.applyColorMap(contrasted_im,cmapy.cmap(cmap))

            frame[np.where(im==0)] = [0,0,0]

            # add scale bar
            frame = scale_bar_xy._add_scale_bar(frame)

            # time stamp
            t = calculate_time_stamp(i, imaging_freq, len_t)
            time_stamp_pos = (0, 0)
            frame = add_time_stamp(frame, time_stamp_pos, t, scale_bar_xy.font)

            # write frame
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def make_comp_ortho_max_video(root, channels, ext='.avi'):

    if len(channels) != 2:
        print(f"Warning: make_comp_ortho_max_video requires exactly two channels, but {len(channels)} were provided. Skipping composite video creation.")
        return
    
    filename = generate_unique_filename('comp_orthomax', ext)

    voxel_dims = channels[0].voxel_dims

    imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    max_z = root['maxz']
    max_y = root['maxy']
    max_x = root['maxx']
    
    len_t, len_z, len_y, len_x = get_projection_dimensions(root)

    # calc scaled Z dimension
    z_to_xy_ratio = voxel_dims[2]/voxel_dims[0]
    scaled_len_z = int(round(len_z*z_to_xy_ratio))
    
    gap = 20

    movie_width = len_x + scaled_len_z + gap
    movie_height = len_y + scaled_len_z + gap

    # calc scale_min, scale_max, and gamma if not provided
    for channel in channels:
        if channel.scale_min == "None" or channel.scale_max == "None":
            channel.scale_min, channel.scale_max = calc_auto_contrast(max_z, channel.n_channel)
        if channel.gamma == "None":
            channel.gamma = 1

    # define scale bars
    scale_bar_length = get_scale_bar_length(root, channel.voxel_dims)
    scale_bar_xy = ScaleBar(
        posY = movie_height,
        posX = len_x,
        length = scale_bar_length,
        pxPerMicron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xy._set_font()
    scale_bar_xz = ScaleBar(
        posY = scaled_len_z,
        posX = len_x + gap + scaled_len_z,
        length = int(len_z*channel.voxel_dims[2]),
        pxPerMicron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xz._set_font()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movie_width,movie_height),1)
    
    try:
        for i in tqdm(range(len_t)):
            processed_channels = []

            for channel in channels:
                im = np.zeros([movie_height,movie_width])

                # copy max projections 
                im_xz = copy.copy(np.flip(max_y[i,channel.n_channel],axis=0))
                im[0:scaled_len_z,0:len_x] = scale_xzyz(im_xz, z_to_xy_ratio)
                im[(scaled_len_z+gap):movie_height,0:len_x] = copy.copy(max_z[i,channel.n_channel,0])
                im_yz = copy.copy(max_x[i,channel.n_channel])
                im[(scaled_len_z+gap):movie_height,(len_x+gap):movie_width] = np.transpose(scale_xzyz(im_yz, z_to_xy_ratio))
                
                contrasted_im = adjust_contrast(im, channel.scale_max, channel.scale_min, channel.gamma)

                # invert channel if specified
                if channel.invert_channel:
                    contrasted_im = 255 - contrasted_im

                processed_channels.append(contrasted_im)

            # Assign colors by channel order
            green_im = processed_channels[0] 
            purple_im = processed_channels[1] 

            # merge images
            frame = cv2.merge((purple_im,green_im,purple_im))
            frame[np.where(im==0)] = [0,0,0]

            # add scale bars
            frame = scale_bar_xy._add_scale_bar(frame)
            frame = scale_bar_xz._add_scale_bar(frame)

            # time stamp
            t = calculate_time_stamp(i, imaging_freq, len_t)
            time_stamp_pos = (0, scaled_len_z+gap)
            frame = add_time_stamp(frame, time_stamp_pos, t, scale_bar_xy.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()

def generate_z_depth_colormap(len_z, cmap):
    #generates a colormap based on z depth, red is the highest z depth, blue is the lowest
    z_depth_colormap = [None]*len_z
    for slice in range(0,len_z):
        z_depth_gray_val = round((slice/len_z)*255)
        z_depth_colormap[slice] = cmapy.color(cmap, z_depth_gray_val)
    return z_depth_colormap

def invert_and_scale(im, invert):
    # invert if rock channel
    if invert:
        im = 255 - im
    scaled_im = np.divide(im,255)
    scaled_im_grayscale = cv2.merge([scaled_im, scaled_im, scaled_im])
    return scaled_im_grayscale

def make_z_depth_ortho_max_video(root, channel, cmap, ext='.avi'):

    filename = generate_unique_filename(channel.name + '_zdepth_orthomax_' + cmap, ext)
    n_channel = channel.n_channel
    voxel_dims = channel.voxel_dims
    scale_max = channel.scale_max
    scale_min = channel.scale_min
    gamma = channel.gamma

    imaging_freq = get_imaging_freq(root.store.path + '/../../parameters.json')

    # TODO: change root to max_projections root
    max_z = root['maxz']
    max_y = root['maxy']
    max_x = root['maxx']
    
    len_t, len_z, len_y, len_x = get_projection_dimensions(root)
    
    # calc scaled Z dimension
    z_to_xy_ratio = voxel_dims[2]/voxel_dims[0]
    scaled_len_z = int(round(len_z*z_to_xy_ratio))

    z_depth_colormap = generate_z_depth_colormap(len_z, cmap)
    z_depth_colormap_xzyz = generate_z_depth_colormap(scaled_len_z, cmap)

    gap = 20

    movie_width = len_x + scaled_len_z + gap
    movie_height = len_y + scaled_len_z + gap

    # calc scale_min, scale_max, and gamma if not provided
    if scale_min == "None" or scale_max == "None":
        scale_min, scale_max = calc_auto_contrast(max_z, n_channel)
    if gamma == "None":
        gamma = 1

    # define scale bars
    scale_bar_length = get_scale_bar_length(root, channel.voxel_dims)
    scale_bar_xy = ScaleBar(
        posY = movie_height,
        posX = len_x,
        length = scale_bar_length,
        pxPerMicron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xy._set_font()
    scale_bar_xz = ScaleBar(
        posY = scaled_len_z,
        posX = len_x + gap + scaled_len_z,
        length = int(len_z*channel.voxel_dims[2]),
        pxPerMicron = 1/channel.voxel_dims[0],
        font = None,
    )
    scale_bar_xz._set_font()

    vid = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,(movie_width,movie_height),1)

    try:
        for i in tqdm(range(len_t)):
            # generate a scaled image for the XY projection
            im_xy = copy.copy(max_z[i,n_channel,0])
            contrasted_im_xy = adjust_contrast(im_xy, scale_max, scale_min, gamma)
            scaled_im_grayscale_xy = invert_and_scale(contrasted_im_xy, channel.invert_channel)
            # apply z depth colormap based on z depths in slice
            z_depths = max_z[i,n_channel,1]
            # TODO: make into functions for XY color assignment and XZ/YZ color assignment
            im_bgr_vals_xy = np.zeros([len_y, len_x, 3]).astype(int)
            for y in range(0,len_y):
                for x in range(0,len_x):
                    z_depth = int(z_depths[y,x])
                    im_bgr_vals_xy[y,x,:] = z_depth_colormap[z_depth]
            frame_xy = np.multiply(scaled_im_grayscale_xy,im_bgr_vals_xy).astype('uint8')

            # generate a scaled image for the XZ projection
            im_xz = copy.copy(max_y[i,n_channel])
            im_xz = scale_xzyz(im_xz, z_to_xy_ratio)

            contrasted_im_xz = adjust_contrast(im_xz, scale_max, scale_min, gamma)
            scaled_im_grayscale_xz = invert_and_scale(contrasted_im_xz, channel.invert_channel)

            # apply z depth colormap based on z depths in slice
            im_bgr_vals_xz = np.zeros([scaled_len_z, len_x, 3]).astype(int)
            for z in range(0,scaled_len_z):
                z_depth = z
                im_bgr_vals_xz[z,:,:] = z_depth_colormap_xzyz[z_depth]
            frame_xz = np.multiply(scaled_im_grayscale_xz,im_bgr_vals_xz).astype('uint8')
            frame_xz = np.flip(frame_xz, axis=0)

            # generate a scaled image for the YZ projection
            im_yz = copy.copy(max_x[i,n_channel])
            im_yz = scale_xzyz(im_yz, z_to_xy_ratio)
            contrasted_im_yz = adjust_contrast(im_yz, scale_max, scale_min, gamma)
            scaled_im_grayscale_yz = invert_and_scale(contrasted_im_yz, channel.invert_channel)

            # apply z depth colormap based on z depths in slice
            im_bgr_vals_yz = np.zeros([scaled_len_z, len_y, 3]).astype(int)
            for z in range(0,scaled_len_z):
                z_depth = z
                im_bgr_vals_yz[z,:,:] = z_depth_colormap_xzyz[z_depth]
            frame_yz = np.multiply(scaled_im_grayscale_yz,im_bgr_vals_yz).astype('uint8')
            frame_yz = np.transpose(frame_yz, (1,0,2))

            # initialize frame 
            frame = np.zeros([movie_height,movie_width,3]).astype('uint8')

            frame[0:scaled_len_z,0:len_x,:] = frame_xz
            frame[(scaled_len_z+gap):movie_height,0:len_x,:] = frame_xy
            frame[(scaled_len_z+gap):movie_height,(len_x+gap):movie_width,:] = frame_yz

            #frame[np.where(scaled_im==0)] = [0,0,0]

            # add scale bars
            frame = scale_bar_xy._add_scale_bar(frame)
            frame = scale_bar_xz._add_scale_bar(frame)

            # time stamp
            t = calculate_time_stamp(i, imaging_freq, len_t)
            time_stamp_pos = (0, scaled_len_z+gap)
            frame = add_time_stamp(frame, time_stamp_pos, t, scale_bar_xy.font)

            # write frame 
            vid.write(frame)

        vid.release()
        cv2.destroyAllWindows()
    except:
        vid.release()
        cv2.destroyAllWindows()