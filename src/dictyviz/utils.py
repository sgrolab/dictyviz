import zarr
import json
from pathlib import Path
import xml.etree.ElementTree as et

from .visualization import Channel


def get_voxel_dims(xml_file, res_lvl=0):
    XML_tree = et.parse(xml_file)
    XML_root = XML_tree.getroot()
    img_metadata = XML_root[res_lvl][0]
    pixel_size_x = float(img_metadata.get('PhysicalSizeX'))
    pixel_size_y = float(img_metadata.get('PhysicalSizeY'))
    pixel_size_z = float(img_metadata.get('PhysicalSizeZ'))
    return pixel_size_x, pixel_size_y, pixel_size_z


def get_axes(zarr_path):
    ome_metadata_path = zarr_path / "OME" / "METADATA.ome.xml"
    if ome_metadata_path.exists():
        pixel_size_x, pixel_size_y, pixel_size_z = get_voxel_dims(ome_metadata_path)
        axes = [
            dict(name='time', type='time', unit='second', scale=1.0),
            dict(name='channel', type='channel', scale=1.0),
            dict(name='z', type='space', unit='micrometer', scale=pixel_size_z),
            dict(name='y', type='space', unit='micrometer', scale=pixel_size_y),
            dict(name='x', type='space', unit='micrometer', scale=pixel_size_x),
        ]
    else:
        axes = None
    return axes


def create_root_store(zarr_file):
    nested_store = zarr.NestedDirectoryStore(zarr_file, dimension_separator='/')
    root = zarr.group(store=nested_store, overwrite=False)

    
def get_channels(json_file):
    with open(json_file) as f:
        channel_specs = json.load(f)["channels"]
    channels = []
    for channel_info in channel_specs:
        current_channel = Channel(name=channel_info["name"],
                                n_channel=channel_info["channelNumber"],
                                scale_max=channel_info["scaleMax"],
                                scale_min=channel_info["scaleMin"],)
        if "gamma" in channel_info:
            current_channel.gamma = channel_info["gamma"]
        if "invertChannel" in channel_info:
            current_channel.invert_channel = channel_info["invertChannel"]
        channels.append(current_channel)
    return channels


def get_cropping_dims(json_file):
    with open(json_file) as f:
        try:
            cropping_params = json.load(f)["croppingParameters"]
            if cropping_params:
                crop_x = cropping_params.get("cropX", [None, None])
                crop_y = cropping_params.get("cropY", [None, None])
                crop_z = cropping_params.get("cropZ", [None, None])
            return [crop_x, crop_y, crop_z]
        except Exception as e:
            return None


def get_slice_depth(json_file):
    with open(json_file) as f:
        movie_specs = json.load(f)["movieSpecs"]
        if "sliceDepth" in movie_specs:
            slice_depth = movie_specs["sliceDepth"]
        else:
            slice_depth = "auto"
    return slice_depth


def get_imaging_freq(json_file):
    with open(json_file) as f:
        imaging_freq = json.load(f)["imagingParameters"]["imagingFrequency"]
    return imaging_freq