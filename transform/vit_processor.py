from transformers import AutoImageProcessor

def get_image_processor(
    pretrained_model_name: str,
    cache_dir: str = None,
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