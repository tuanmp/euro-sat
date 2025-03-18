from transformers import BaseImageProcessor, AutoImageProcessor

def get_image_processor(
    resample = None,
    image_mean = None,
    image_std = None,
    size = None,
    default_to_square = True,
    crop_size = None,
    do_resize = None,
    do_center_crop = None,
    do_rescale = None,
    rescale_factor = 1 / 255,
    do_normalize = None,
    do_convert_rgb = None,
    model_input_names = ["pixel_values"], **kwargs): 
    """
    Get an image processor for a given model.
    Args:
        resample: Resampling filter.
        image_mean: Mean for normalization.
        image_std: Standard deviation for normalization.
        size: Size of the image.
        default_to_square: Whether to default to square images.
        crop_size: Size of the crop.
        do_resize: Whether to resize the image.
        do_center_crop: Whether to center crop the image.
        do_rescale: Whether to rescale the image.
        rescale_factor: Factor for rescaling.
        do_normalize: Whether to normalize the image.
        do_convert_rgb: Whether to convert the image to RGB.
        model_input_names: Input names for the model."""

    return BaseImageProcessor(
        resample=resample,
        image_mean=image_mean,
        image_std=image_std,
        size=size,
        default_to_square=default_to_square,
        crop_size=crop_size,
        do_resize=do_resize,
        do_center_crop=do_center_crop,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        do_normalize=do_normalize,
        do_convert_rgb=do_convert_rgb,
        model_input_names=model_input_names
    )

def get_pretrained_processor(pretrained_model_name: str,
    cache_dir: str = "./cache",
    use_fast: bool = True
) -> AutoImageProcessor:
    """
    Get an image processor for a given model.
    Args:
        pretrained_model_name: The name of the pretrained model.
        cache_dir: The cache directory.
        use_fast: Whether to use the fast version of the processor.
    Returns:
        The image processor.
    """
    return AutoImageProcessor.from_pretrained(
        pretrained_model_name,
        cache_dir=cache_dir,
        use_fast=use_fast
    )

