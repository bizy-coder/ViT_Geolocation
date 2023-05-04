import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
import fastai
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
import open_clip
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations as A
import PIL


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

config = SimpleNamespace(
    batch_size=32,
    img_size=224,
    seed=42,
    pretrained=True,
    normalize=True,
    model_name="vit_small_patch32_224",
    epochs=20,  # FIX THIS LATER
    learning_rate=2e-3,
    resize="crop",
)


def albumentations_transforms(x: PIL.Image.Image, img_size=224):
    transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=99, quality_upper=100),
            A.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7
            ),
            A.Resize(img_size, img_size),
            A.Cutout(
                max_h_size=int(img_size * 0.4),
                max_w_size=int(img_size * 0.4),
                num_holes=1,
                p=0.5,
            ),
        ]
    )

    return PIL.Image.fromarray(transforms(image=np.array(x))["image"])


def load_data(augment=False, config=config, test_set=False):
    if test_set:
        return load_test_data(config, augment)

    resize_method = (
        ResizeMethod.Squish if config.resize == "squish" else ResizeMethod.Crop
    )

    if augment:
        dls = fastai.vision.data.ImageDataLoaders.from_folder(
            "code/data/",
            valid_pct=0.2,
            seed=config.seed,
            bs=config.batch_size,
            item_tfms=[
                Resize(config.img_size, method=resize_method),
                albumentations_transforms,
            ],
        )
    else:
        dls = fastai.vision.data.ImageDataLoaders.from_folder(
            "code/data/",
            valid_pct=0.2,
            seed=config.seed,
            bs=config.batch_size,
            item_tfms=[Resize(config.img_size, method=resize_method)],
        )

    mean, std = (0.48145466, 0.4578275, 0.40821073), (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    if config.normalize:
        dls.add_tfms([Normalize.from_stats(mean, std)], "after_batch")

    return dls


def load_test_data(config=config, augment=False):
    resize_method = (
        ResizeMethod.Squish if config.resize == "squish" else ResizeMethod.Crop
    )

    if augment:
        dls = fastai.vision.data.ImageDataLoaders.from_folder(
            "code/test_data/",
            valid_pct=0.99,
            seed=config.seed,
            bs=config.batch_size,
            item_tfms=[
                Resize(config.img_size, method=resize_method),
                albumentations_transforms,
            ],
        )
    else:
        dls = fastai.vision.data.ImageDataLoaders.from_folder(
            "code/test_data/",
            valid_pct=0.99,
            seed=config.seed,
            bs=config.batch_size,
            item_tfms=[Resize(config.img_size, method=resize_method)],
        )

    mean, std = (0.48145466, 0.4578275, 0.40821073), (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    if config.normalize:
        dls.add_tfms([Normalize.from_stats(mean, std)], "after_batch")

    return dls


def create_learner(model_name, dls=None, augment=False, test_set=False, config=config):
    if dls is None:
        dls = load_data(augment, config, test_set)

    top_2_accuracy = functools.partial(top_k_accuracy, k=2)
    top_3_accuracy = functools.partial(top_k_accuracy, k=3)
    top_5_accuracy = functools.partial(top_k_accuracy, k=5)
    top_10_accuracy = functools.partial(top_k_accuracy, k=10)
    learn = vision_learner(
        dls,
        model_name,
        metrics=[
            accuracy,
            top_2_accuracy,
            top_3_accuracy,
            top_5_accuracy,
            top_10_accuracy,
        ],
        concat_pool=True,
    ).to_fp16()

    # To device
    learn.dls.device = device

    return learn


def load_local_model(path, learn):
    file = torch.load(path, map_location=device)
    learn.model.load_state_dict(file)

    return learn


def load_wandb_model(model_num, learn):
    run = wandb.init()
    artifact = run.use_artifact(f"ben_z/geolocation/model:v{model_num}", type="model")
    artifact_dir = artifact.download()
    file = torch.load(os.path.join(artifact_dir, "model.pth"), map_location=device)

    # create a learner with vit_small_patch32_224
    learn.model.load_state_dict(file)

    # Move model to GPU
    learn.model = learn.model.to(device)
    learn.dls.device = device

    return learn


def get_predictions(learn):
    # Predict on validation set
    preds, targs = learn.get_preds()

    return preds, targs


def display_confusion_matrix(preds, targs, states, save_path=None):
    # Get predictions and targets
    if isinstance(preds, torch.Tensor):
        preds, targs = preds.numpy(), targs.numpy()

    # Get top 1 predictions
    top1_preds = np.argmax(preds, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(targs, top1_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Add label of what the numbers mean
    tick_marks = np.arange(len(states))
    plt.xticks(tick_marks, states, rotation=90)
    plt.yticks(tick_marks, states, rotation=0)
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return cm


def get_most_confused(cm, states):
    # Get most confused states
    most_confused = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                most_confused.append((states[i], states[j], cm[i][j]))

    # Sort by most confused
    most_confused.sort(key=lambda x: x[2], reverse=True)

    return most_confused


def get_top_k_accuracy(preds, targs, k):
    # Get predictions and targets
    if isinstance(preds, torch.Tensor):
        preds, targs = preds.numpy(), targs.numpy()

    # Get top k predictions
    topk_preds = np.argsort(preds, axis=1)[:, -k:]

    # Get top 1 predictions
    top1_preds = np.argmax(preds, axis=1)

    # Get top k accuracy
    topk_correct = 0
    for i in range(len(preds)):
        if targs[i] in topk_preds[i]:
            topk_correct += 1

    # Get top 1 accuracy
    top1_correct = 0
    for i in range(len(preds)):
        if targs[i] == top1_preds[i]:
            top1_correct += 1

    return topk_correct / len(preds), top1_correct / len(preds)


def plot_top_k_accuracy(preds, targs, save_path=None):
    # Get predictions and targets
    if isinstance(preds, torch.Tensor):
        preds, targs = preds.numpy(), targs.numpy()

    # Get top k accuracy for k = 1, 2, 3, 5, 10
    topk_accs = []
    for k in range(1, 51):
        topk_accs.append(get_top_k_accuracy(preds, targs, k))

    # Plot top k accuracy
    plt.figure(figsize=(10, 10))
    plt.plot(list(range(1, 51)), topk_accs)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.xticks(list(range(1, 51, 2)))
    plt.title("Top k Accuracy")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return topk_accs


def plot_class_precision_and_recall(cm, states, save_path=None):
    # Get precision and recall for each class, no built-in function for this
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)

    # Sort states by precision
    states = [x for _, x in sorted(zip(precision, states))]
    recall = [x for _, x in sorted(zip(precision, recall))]
    precision.sort()

    # Plot precision and recall
    plt.figure(figsize=(15, 15))
    # Place precision and recall side by side
    plt.bar(np.arange(len(states)) - 0.15, recall, width=0.3, label="Recall")
    plt.bar(np.arange(len(states)) + 0.15, precision, width=0.3, label="Precision")

    # Add labels
    plt.xticks(np.arange(len(states)), states, rotation=90)

    plt.xlabel("State")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.title("Precision and Recall")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_class_f1(cm, states, save_path=None):
    # Get precision and recall for each class, no built-in function for this
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.diag(cm) / np.sum(cm, axis=0)

    # Calculate f1 score
    f1 = 2 * (precision * recall) / (precision + recall)

    # Sort states by f1 score
    states = [x for _, x in sorted(zip(f1, states))]
    f1.sort()

    # Plot f1 score
    plt.figure(figsize=(15, 15))
    plt.bar(np.arange(len(states)), f1, width=0.6)

    # Add labels
    plt.xticks(np.arange(len(states)), states, rotation=90)

    plt.xlabel("State")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.title("F1 Score")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
