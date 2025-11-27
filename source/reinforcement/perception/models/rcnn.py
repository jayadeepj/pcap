import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights


def get_kp_model(num_keypoints, num_classes, weights_path=None):
    """
    Get a keypoint RCNN model.

    Args:
        num_keypoints (int): Number of keypoints for keypoint detection.
        num_classes (int): Number of classes, including the background class.
        weights_path (str, optional): Path to pre-trained weights file.

    Returns:
        torchvision.models.detection.KeyPointRCNN: The keypoint RCNN model.
    """
    # Input validation
    assert num_keypoints > 0, "num_keypoints must be a positive integer."
    assert num_classes > 1, "num_classes must be a positive integer greater than 1."

    # Anchor generator configuration
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)
    )

    # Create the keypoint RCNN model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_keypoints=num_keypoints,
        num_classes=num_classes,  # Background is the first class, object is the second class
        rpn_anchor_generator=anchor_generator
    )

    # Load pre-trained weights if available
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

        # Check if the loaded weights match the model architecture
        assert num_keypoints == model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels, \
            "Number of keypoints in loaded weights does not match the specified model."

        # model.half()  # drop precision to fp16 to gain speed

    return model
