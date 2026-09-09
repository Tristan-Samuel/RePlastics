from pathlib import Path
import argparse
import copy
import os

import matplotlib

if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"], force=True)
elif os.environ.get("COLAB_RELEASE_TAG") or os.environ.get(
    "COLAB_BACKEND_VERSION"
):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
)

from trashnet import (
    EXPECTED_CLASSES,
    assert_split_layout,
    split_dirs_for,
)


DEFAULT_MODEL_PATH = Path("models/resnext50_metal_plastic.pt")
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNeXt-50 on metal vs plastic."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load the saved model and continue training "
            "to improve it."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help=(
            "Dataset root with train/validation/test "
            "class folders. Use data_real for in-the-wild "
            "photos."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Checkpoint path to load and save.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the classification head.",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help=(
            "Learning rate for unfrozen backbone stages. "
            "Defaults to --lr / 10."
        ),
    )
    parser.add_argument(
        "--unfreeze",
        choices=("fc", "layer4", "layer3", "all"),
        default="fc",
        help=(
            "Which stages to train. fc = head only; "
            "layer4 also trains the last residual stage; "
            "layer3 adds the stage before that; "
            "all trains the whole network."
        ),
    )
    return parser.parse_args()


args = parse_args()

if args.lr_backbone is None:
    args.lr_backbone = args.lr / 10

MODEL_PATH = args.model
HEADLESS = matplotlib.get_backend().lower() == "agg"


def show_plot():
    if HEADLESS:
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# Pick GPU / Apple Silicon / CPU
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}", flush=True)
print(f"Data dir: {args.data_dir}", flush=True)
print(f"Unfreeze: {args.unfreeze}", flush=True)


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

weights = ResNeXt50_32X4D_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# For validation and testing, don't randomly modify the images.
eval_transform = weights.transforms()


# ---------------------------------------------------------
# Load dataset from train / validation / test folders
# ---------------------------------------------------------

assert_split_layout(args.data_dir)
split_dirs = split_dirs_for(args.data_dir)

training_images = datasets.ImageFolder(
    split_dirs["train"],
    transform=train_transform,
)

validation_images = datasets.ImageFolder(
    split_dirs["validation"],
    transform=eval_transform,
)

test_images = datasets.ImageFolder(
    split_dirs["test"],
    transform=eval_transform,
)

print("Classes:", training_images.classes, flush=True)
print("Class mapping:", training_images.class_to_idx, flush=True)

assert training_images.classes == EXPECTED_CLASSES, (
    "Each split folder should contain only "
    "'metal' and 'plastic' folders."
)
assert validation_images.classes == EXPECTED_CLASSES
assert test_images.classes == EXPECTED_CLASSES
assert (
    training_images.class_to_idx
    == validation_images.class_to_idx
    == test_images.class_to_idx
)


train_loader = DataLoader(
    training_images,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_images,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_images,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"Training images:   {len(training_images)}", flush=True)
print(f"Validation images: {len(validation_images)}", flush=True)
print(f"Test images:       {len(test_images)}", flush=True)


# ---------------------------------------------------------
# Create ResNeXt-50
# ---------------------------------------------------------

model = resnext50_32x4d(
    weights=None if args.resume else weights
)


# ResNeXt normally outputs 1000 ImageNet classes.
#
# Replace its final layer with:
#
#     metal
#     plastic
#
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2,
)

model = model.to(device)

if args.resume:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot resume: {MODEL_PATH} was not found."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    saved_class_to_idx = checkpoint.get("class_to_idx")

    if saved_class_to_idx != training_images.class_to_idx:
        raise ValueError(
            "Saved model class mapping does not match "
            "the current dataset."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(f"Resumed from {MODEL_PATH}", flush=True)


def apply_unfreeze(model, unfreeze):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    if unfreeze in {"layer4", "layer3", "all"}:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    if unfreeze in {"layer3", "all"}:
        for parameter in model.layer3.parameters():
            parameter.requires_grad = True

    if unfreeze == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True


def build_optimizer(model, args):
    param_groups = [
        {
            "params": list(model.fc.parameters()),
            "lr": args.lr,
        },
    ]

    if args.unfreeze == "layer4":
        param_groups.append(
            {
                "params": list(model.layer4.parameters()),
                "lr": args.lr_backbone,
            }
        )
    elif args.unfreeze == "layer3":
        param_groups.append(
            {
                "params": list(model.layer4.parameters()),
                "lr": args.lr_backbone,
            }
        )
        param_groups.append(
            {
                "params": list(model.layer3.parameters()),
                "lr": args.lr_backbone / 10,
            }
        )
    elif args.unfreeze == "all":
        backbone = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not name.startswith("fc.")
        ]
        param_groups.append(
            {
                "params": backbone,
                "lr": args.lr_backbone,
            }
        )

    return torch.optim.AdamW(param_groups)


apply_unfreeze(model, args.unfreeze)

trainable = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]
print(
    f"Trainable tensors: {len(trainable)} "
    f"(head lr={args.lr}, backbone lr={args.lr_backbone})",
    flush=True,
)


# ---------------------------------------------------------
# Loss function + optimizer
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = build_optimizer(model, args)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def create_confusion_matrix(
    model,
    data_loader,
    class_names,
    device,
):
    model.eval()

    true_labels = []
    predicted_labels = []
    mistakes = []
    sample_index = 0
    dataset = data_loader.dataset

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                if predicted_label != true_label:
                    image_path = dataset.samples[
                        sample_index + i
                    ][0]

                    mistakes.append({
                        "path": image_path,
                        "true": class_names[true_label],
                        "predicted": class_names[
                            predicted_label
                        ],
                        "confidence": probabilities[
                            i, predicted_label
                        ].item(),
                    })

            sample_index += labels.size(0)

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )

    print("Confusion matrix:", flush=True)
    print(cm, flush=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        values_format="d",
    )

    plt.title("Validation Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=200,
    )

    show_plot()

    mistakes.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    print()

    if not mistakes:
        print("No misclassified validation images.", flush=True)
        return

    print(
        f"Misclassified validation images "
        f"({len(mistakes)}):",
        flush=True,
    )

    report_path = Path("misclassified_validation.txt")

    with report_path.open("w") as report:
        for mistake in mistakes:
            line = (
                f"{mistake['path']} | "
                f"true: {mistake['true']} | "
                f"pred: {mistake['predicted']} "
                f"({mistake['confidence']:.2%})"
            )
            print(line, flush=True)
            report.write(line + "\n")

    print(f"Wrote {report_path}", flush=True)

    columns = min(4, len(mistakes))
    rows = (
        len(mistakes) + columns - 1
    ) // columns

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4.4 * rows),
    )

    if rows == 1 and columns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for index, mistake in enumerate(mistakes):
        axis = axes[index]
        image = ImageOps.exif_transpose(
            Image.open(mistake["path"])
        ).convert("RGB")
        axis.imshow(image)
        axis.set_title(
            f"{mistake['true']} → "
            f"{mistake['predicted']}\n"
            f"{mistake['confidence']:.1%}  "
            f"{Path(mistake['path']).name}"
        )
        axis.axis("off")

    for axis in axes[len(mistakes):]:
        axis.axis("off")

    figure.suptitle(
        "Misclassified validation images "
        "(most confident errors first)"
    )
    figure.tight_layout()

    figure.savefig(
        "misclassified_validation.png",
        dpi=200,
    )

    show_plot()


def save_checkpoint(test_accuracy=None):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": best_model_weights,
        "class_to_idx": training_images.class_to_idx,
        "best_validation_accuracy": best_validation_accuracy,
    }

    if test_accuracy is not None:
        payload["test_accuracy"] = test_accuracy

    torch.save(payload, MODEL_PATH)


best_validation_accuracy = 0.0
best_model_weights = copy.deepcopy(
    model.state_dict()
)

if args.resume:
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            predicted_classes = outputs.argmax(dim=1)

            correct_predictions += (
                predicted_classes == labels
            ).sum().item()

            total_predictions += labels.size(0)

    best_validation_accuracy = (
        correct_predictions
        / total_predictions
    )

    print(
        f"Starting validation loss: "
        f"{running_loss / total_predictions:.4f} | "
        f"Starting validation accuracy: "
        f"{best_validation_accuracy:.2%}",
        flush=True,
    )


for epoch in range(args.epochs):

    # =====================================================
    # TRAINING
    # =====================================================

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_index, (images, labels) in enumerate(
        train_loader,
        start=1,
    ):

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous iteration.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)

        # Compare predictions against correct labels.
        loss = criterion(outputs, labels)

        # Calculate gradients.
        loss.backward()

        # Update the trainable parameters.
        optimizer.step()

        # Statistics.
        running_loss += (
            loss.item() * images.size(0)
        )

        predicted_classes = outputs.argmax(dim=1)

        correct_predictions += (
            predicted_classes == labels
        ).sum().item()

        total_predictions += labels.size(0)

        if batch_index % 20 == 0 or batch_index == len(
            train_loader
        ):
            print(
                f"  Epoch {epoch + 1} "
                f"batch {batch_index}/{len(train_loader)} "
                f"loss {loss.item():.4f}",
                flush=True,
            )


    train_loss = (
        running_loss / total_predictions
    )

    train_accuracy = (
        correct_predictions
        / total_predictions
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0


    # We don't need gradients when evaluating.
    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += (
                loss.item() * images.size(0)
            )

            predicted_classes = outputs.argmax(dim=1)

            correct_predictions += (
                predicted_classes == labels
            ).sum().item()

            total_predictions += labels.size(0)


    validation_loss = (
        running_loss / total_predictions
    )

    validation_accuracy = (
        correct_predictions
        / total_predictions
    )


    # =====================================================
    # Print results
    # =====================================================

    print(
        f"Epoch {epoch + 1}/{args.epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Val loss: {validation_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.2%} | "
        f"Validation accuracy: "
        f"{validation_accuracy:.2%}",
        flush=True,
    )


    # Keep the best version of the model, and write it
    # to disk immediately so a crash or Ctrl+C does not
    # lose the weights.
    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = (
            validation_accuracy
        )

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        save_checkpoint()

        print(
            f"Saved best model so far to {MODEL_PATH} "
            f"({best_validation_accuracy:.2%} val)",
            flush=True,
        )


# ---------------------------------------------------------
# Test the best model on held-out images
# ---------------------------------------------------------

model.load_state_dict(best_model_weights)
model.eval()

correct_predictions = 0
total_predictions = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predicted_classes = outputs.argmax(dim=1)

        correct_predictions += (
            predicted_classes == labels
        ).sum().item()

        total_predictions += labels.size(0)


test_accuracy = (
    correct_predictions
    / total_predictions
)


# ---------------------------------------------------------
# Save best model
# ---------------------------------------------------------

save_checkpoint(test_accuracy=test_accuracy)

print()
print(
    f"Best validation accuracy: "
    f"{best_validation_accuracy:.2%}",
    flush=True,
)
print(
    f"Test accuracy: "
    f"{test_accuracy:.2%}",
    flush=True,
)

print(f"Saved model to {MODEL_PATH}", flush=True)

create_confusion_matrix(
    model,
    validation_loader,
    training_images.classes,
    device,
)
