import torch
import tkinter as tk

from pathlib import Path
from model import CNNClassifier
from PIL import Image, ImageDraw
from torchvision import transforms

device = torch.device("cpu")

model = CNNClassifier().to(device)

model_path = Path(__file__).resolve().parent / "mnist_cnn.pth"

model.load_state_dict(
    torch.load(model_path, map_location=device, weights_only=True)
)

model.eval()

root = tk.Tk()
root.title("MNIST Digit Recognizer")
root.geometry("780x780")
root.configure(bg="#1e1e1e")

title_label = tk.Label(
    root,
    text="Handwritten Digit Recognizer",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#1e1e1e"
)

title_label.pack(pady=15)

canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black"
)

canvas.pack(pady=20)

image = Image.new("L", (280, 280), color=0)
image_draw = ImageDraw.Draw(image)

def draw(event):
    x = event.x
    y = event.y

    radius = 10

    # Draw on the Tkinter canvas
    canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="white",
        outline="white"
    )

    # Draw on the Pillow image
    image_draw.ellipse(
        (
            x - radius,
            y - radius,
            x + radius,
            y + radius
        ),
        fill=255
    )

canvas.bind("<Button-1>", draw)
canvas.bind("<B1-Motion>", draw)

def clear_canvas():
    canvas.delete("all")

    for bar in bars:
        bar.delete("all")

    for label in percentage_labels:
        label.config(text="0.0%")

    # Reset/Change the Pillow image to black
    image_draw.rectangle(
        (0, 0, 280, 280),
        fill=0
    )

    prediction_label.config(
        text="Draw a digit and click Predict"
    )

button_frame = tk.Frame(
    root,
    bg="#1e1e1e"
)

button_frame.pack(pady=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    width=12,
    bg="#404040",
    fg="white"
)

clear_button.pack(side="left", padx=5)

def prepare_image():
    # Find the boundaries of the drawing
    bbox = image.getbbox()

    if bbox is None:
        return None

    # Crop away empty black space
    cropped = image.crop(bbox)

    # Resize while preserving proportions
    width, height = cropped.size

    scale = 20 / max(width, height)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create a new black 28x28 image
    processed = Image.new("L", (28, 28), color=0)

    # Center the digit
    x = (28 - new_width) // 2
    y = (28 - new_height) // 2

    processed.paste(resized, (x, y))

    # Convert to a tensor
    tensor = transforms.ToTensor()(processed)

    return tensor.unsqueeze(0)

def predict_digit():
    tensor = prepare_image()

    if tensor is None:
        prediction_label.config(
            text="Please draw a digit first!"
        )
        return

    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = probabilities.argmax(dim=1)
        confidence = probabilities[0, prediction.item()].item() * 100

    update_bars(probabilities[0])

    prediction_label.config(
        text=f"Prediction: {prediction.item()}\nConfidence: {confidence:.2f}%"
    )

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    width=12,
    bg="#3478C8",
    fg="white"
)

predict_button.pack(side="left", padx=5)

prediction_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 14, "bold"),
    fg="white",
    bg="#1e1e1e"
)

prediction_label.pack(pady=10)

bars = []
percentage_labels = []

for digit in range(10):
    row = tk.Frame(
        root,
        bg="#1e1e1e"
    )
    row.pack()

    label = tk.Label(
        row,
        text=str(digit),
        width=3,
        fg="white",
        bg="#1e1e1e"
    )

    label.pack(side="left")

    bar = tk.Canvas(
        row,
        width=200,
        height=15,
        bg="lightgray",
        highlightthickness=0
    )

    bar.pack(side="left")

    bars.append(bar)

    percentage_label = tk.Label(
        row,
        text="0.0%",
        width=7,
        fg="white",
        bg="#1e1e1e"
    )

    percentage_label.pack(side="left", padx=5)
    percentage_labels.append(percentage_label)

def update_bars(probabilities):
    for digit in range(10):
        probability = probabilities[digit].item()

        bar = bars[digit]
        bar.delete("all")

        bar.create_rectangle(
            0, 0,
            probability * 200, 15,
            fill="green",
            outline=""
        )

        percentage_labels[digit].config(
            text=f"{probability * 100:.1f}%"
        )

root.mainloop()