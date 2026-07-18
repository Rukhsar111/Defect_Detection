import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
from ultralytics import RTDETR


# ============================================================================
# YOLO SETUP
# ============================================================================
# Model is no longer loaded automatically at startup - use the "Load Model"
# button in the UI to pick a .pt or .onnx file (best.pt, best_FP32.onnx, etc.)
model = None
current_model_path = None

# Single-class setup: only "Defect"
Object_classes = ["Defect"]
Object_colors = [
    (0, 0, 255)        # Defect = red (BGR format)
]

class_conf = {
    "Defect": 0.5
}


# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
current_image_path = None
show_labels = False
show_masks = False


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================
def resize_image_maintain_aspect(image, max_width=700, max_height=400):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    aspect = width / height

    if width > max_width or height > max_height:
        if aspect > 1:  # Wider than tall
            new_width = min(width, max_width)
            new_height = int(new_width / aspect)
        else:  # Taller than wide
            new_height = min(height, max_height)
            new_width = int(new_height * aspect)
    else:
        new_width = width
        new_height = height

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def process_image(image_path):
    """Runs YOLO detection → draws boxes/masks → returns frames and counts"""
    if model is None:
        messagebox.showerror("Error", "No model loaded. Please load a model first.")
        return None, None, None, None

    # Read original image
    original_frame = cv2.imread(image_path)

    if original_frame is None:
        messagebox.showerror("Error", "Failed to load image")
        return None, None, None, None

    print(f"Image dimensions: {original_frame.shape}")

    # Create a copy for processing
    processed_frame = original_frame.copy()

    # Run YOLO detection (timed for per-frame latency reporting)
    inference_start = time.perf_counter()
    results = model(image_path, iou=0.5)[0]
    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    object_counts = {"Defect": 0}

    # Check if masks are available
    has_masks = hasattr(results, 'masks') and results.masks is not None

    # Process each detection
    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls)
        label = Object_classes[cls_id]
        score = float(box.conf)

        # Skip if confidence below threshold
        if score < class_conf[label]:
            continue

        object_counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = Object_colors[cls_id]

        # Draw masks if enabled and available
        if show_masks and has_masks:
            try:
                if idx < len(results.masks.data):
                    # Get mask for this detection
                    mask = results.masks.data[idx].cpu().numpy()

                    # Resize mask to match image dimensions
                    mask_resized = cv2.resize(mask, (processed_frame.shape[1], processed_frame.shape[0]))

                    # Create colored overlay
                    mask_bool = mask_resized > 0.5
                    overlay = processed_frame.copy()
                    overlay[mask_bool] = color

                    # Blend overlay with original (40% mask, 60% original)
                    cv2.addWeighted(overlay, 0.4, processed_frame, 0.6, 0, processed_frame)

            except Exception as e:
                print(f"Error drawing mask for detection {idx}: {e}")

        # Draw bounding box
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 3)

        # Draw labels if enabled
        if show_labels:
            label_text = f"{label} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Draw background rectangle for text
            cv2.rectangle(processed_frame, (x1, y1 - text_height - baseline - 10), (x1 + text_width + 10, y1), color, -1)

            # Draw text
            cv2.putText(processed_frame, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return original_frame, processed_frame, object_counts, inference_latency_ms



# ============================================================================
# UI CALLBACK FUNCTIONS
# ============================================================================
def load_model():
    """Handle model loading from the UI (.pt or .onnx weights)"""
    file_path = filedialog.askopenfilename(
        title="Select a Model File",
        filetypes=[
            ("YOLO Weights", "*.pt *.onnx"),
            ("PyTorch Weights", "*.pt"),
            ("ONNX Model", "*.onnx"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        global model, current_model_path

        model_label.config(text=f"⏳ Loading {Path(file_path).name} ...", fg="#F57C00")
        root.update_idletasks()  # force UI to repaint the loading message immediately

        loaded_model = YOLO(file_path)

        # Sanity check: run a tiny warmup so a bad/incompatible file fails now,
        # not on the user's first Detect click.
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        loaded_model.predict(dummy, verbose=False)

        model = loaded_model
        current_model_path = file_path

        model_label.config(
            text=f"✅ Model loaded: {Path(file_path).name}",
            fg="#2e7d32"
        )

        # Only enable Detect if an image has already been uploaded too
        if current_image_path:
            detect_btn.config(state=tk.NORMAL)

    except Exception as e:
        model = None
        current_model_path = None
        model_label.config(text="❌ Failed to load model", fg="#c62828")
        detect_btn.config(state=tk.DISABLED)
        messagebox.showerror("Model Load Error", f"Could not load model:\n{str(e)}")


def upload_image():
    """Handle image upload"""
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[
            ("Image Files", "*.jpg *.png *.jpeg *.bmp *.gif"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        global current_image_path
        current_image_path = file_path

        # Read and display original image
        original_frame = cv2.imread(file_path)

        if original_frame is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        # Convert BGR to RGB
        original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_resized = resize_image_maintain_aspect(original_pil)
        original_tk = ImageTk.PhotoImage(original_resized)

        # Update original image display
        original_label.config(image=original_tk, text="")
        original_label.image = original_tk

        # Clear predicted side
        predicted_label.config(text="Click 'Detect' to run detection", image="")
        latency_label.config(text="")

        # Update status
        count_label.config(
            text="Image uploaded successfully! Click 'Detect' to analyze",
            fg="#1976D2",
            font=("Arial", 16)
        )

        # Enable detect button only if a model is already loaded too
        if model is not None:
            detect_btn.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def detect_objects():
    """Handle defect detection"""
    if model is None:
        messagebox.showwarning("Warning", "Please load a model first!")
        return

    if not current_image_path:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    try:
        # Run YOLO detection
        original_frame, processed_frame, object_counts, inference_latency_ms = process_image(current_image_path)

        if processed_frame is None:
            return

        # Convert BGR to RGB
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)
        processed_resized = resize_image_maintain_aspect(processed_pil)
        processed_tk = ImageTk.PhotoImage(processed_resized)

        # Update predicted image display
        predicted_label.config(image=processed_tk, text="")
        predicted_label.image = processed_tk

        # Update count display
        total_count = object_counts["Defect"]

        if total_count == 0:
            count_text = "✅ No Defects Detected"
            text_color = "#2e7d32"
        else:
            count_text = f"⚠️ Total Defects Detected: {total_count}"
            text_color = "#c62828"

        count_label.config(
            text=count_text,
            fg=text_color,
            font=("Arial", 18, "bold")
        )

        # Update latency display
        fps = 1000.0 / inference_latency_ms if inference_latency_ms > 0 else 0.0
        latency_label.config(
            text=f"⏱️ Inference Latency: {inference_latency_ms:.1f} ms  |  ~{fps:.1f} FPS",
            fg="#455A64"
        )

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Debug - Error details: {e}")


def toggle_show_labels():
    """Toggle label display"""
    global show_labels
    show_labels = bool(show_var.get())
    state_text = "ON" if show_labels else "OFF"
    print(f"Show Labels: {state_text}")


def toggle_show_masks():
    """Toggle mask display"""
    global show_masks
    show_masks = bool(show_masks_var.get())
    state_text = "ON" if show_masks else "OFF"
    print(f"Show Masks: {state_text}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================
if __name__ == "__main__":

    # Create main window
    root = tk.Tk()
    root.title("Defect Detection Application")
    root.geometry("1600x900")
    root.config(background="lightgreen")

    # ========================================================================
    # HEADING
    # ========================================================================
    heading_label = tk.Label(root, text="Defect Detection Demo", font=("Arial", 44, "bold"), bg="white")
    heading_label.pack(pady=10)


    # ========================================================================
    # IMAGE DISPLAY SECTION
    # ========================================================================
    images_container = tk.Frame(root, bg="white")
    images_container.pack(pady=0, padx=20, fill=tk.BOTH, expand=True)

    # Left Frame - Original Image
    left_frame = tk.Frame(images_container, bg="white")
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

    left_title = tk.Label(left_frame, text="Original Image", font=("Arial", 20, "bold"), bg="white", fg="#388E3C")
    left_title.pack(pady=10)

    original_label = tk.Label(left_frame, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=3, text="No image uploaded", font=("Arial", 14), fg="#999999")
    original_label.pack(fill=tk.BOTH, expand=True)

    # Right Frame - Predicted Result
    right_frame = tk.Frame(images_container, bg="white")
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

    # Title row with Predicted Result label on the left and model controls on the right
    right_title_row = tk.Frame(right_frame, bg="white")
    right_title_row.pack(fill=tk.X, pady=10)

    right_title = tk.Label(right_title_row, text="Predicted Result", font=("Arial", 20, "bold"), bg="white", fg="#1976D2")
    right_title.pack(side=tk.LEFT)

    model_control_frame = tk.Frame(right_title_row, bg="white")
    model_control_frame.pack(side=tk.RIGHT)

    model_label = tk.Label(model_control_frame, text="⚠️ No model loaded", font=("Arial", 9, "bold"), bg="white", fg="#c62828")
    model_label.pack(side=tk.RIGHT, padx=(8, 0))

    load_model_btn = tk.Button(model_control_frame, text="🧠 Load Model", font=("Arial", 9, "bold"), bg="#8E24AA", fg="white", activebackground="#6A1B9A", activeforeground="white", padx=14, pady=4, command=load_model, cursor="hand2")
    load_model_btn.pack(side=tk.RIGHT)

    predicted_label = tk.Label(right_frame, bg="#f0f0f0", relief=tk.SUNKEN, borderwidth=3, text="No image uploaded", font=("Arial", 14), fg="#999999")
    predicted_label.pack(fill=tk.BOTH, expand=True)


    # ========================================================================
    # COUNT DISPLAY
    # ========================================================================
    count_frame = tk.Frame(root, bg="white")
    count_frame.pack(pady=10)

    count_label = tk.Label(count_frame, text="Upload an image to start detection", font=("Arial", 16), bg="white", fg="#666666")
    count_label.pack()

    latency_label = tk.Label(count_frame, text="", font=("Arial", 12, "bold"), bg="white", fg="#455A64")
    latency_label.pack(pady=(4, 0))

    # ========================================================================
    # BOTTOM SECTION: BUTTONS AND THRESHOLD
    # ========================================================================
    root_bg = root.cget("background")

    bottom_container = tk.Frame(root, bg=root_bg)
    bottom_container.pack(pady=5, fill=tk.X, padx=20)

    # ------------------------------------------------------------------------
    # LEFT SIDE: BUTTONS AND CHECKBOXES
    # ------------------------------------------------------------------------
    button_frame = tk.Frame(bottom_container, bg=root_bg)
    button_frame.pack(side=tk.LEFT, padx=(650, 0), pady=(0, 50))

    # Checkboxes Frame (side by side)
    checkbox_frame = tk.Frame(button_frame, bg=root_bg)
    checkbox_frame.pack(pady=(0, 10))

    # Show Labels Checkbox
    show_var = tk.BooleanVar(value=show_labels)
    show_labels_cb = tk.Checkbutton(checkbox_frame, text="Show Labels", variable=show_var, onvalue=True, offvalue=False, command=toggle_show_labels, font=("Arial", 10, "bold"), bg=root_bg, cursor="hand2")
    show_labels_cb.pack(side=tk.LEFT, padx=10)

    # Show Masks Checkbox
    show_masks_var = tk.BooleanVar(value=show_masks)
    show_masks_cb = tk.Checkbutton(checkbox_frame, text="Show Masks", variable=show_masks_var, onvalue=True, offvalue=False, command=toggle_show_masks, font=("Arial", 10, "bold"), bg=root_bg, cursor="hand2")
    show_masks_cb.pack(side=tk.LEFT, padx=10)

    # Upload Button
    upload_btn = tk.Button(button_frame, text="📁 Upload Image", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white", padx=40, pady=12, command=upload_image, cursor="hand2")
    upload_btn.pack(pady=5)

    # Detect Button
    detect_btn = tk.Button(button_frame, text="🔍 Detect Defects", font=("Arial", 10, "bold"), bg="#2196F3", fg="white", activebackground="#1976D2", activeforeground="white", padx=40, pady=12, command=detect_objects, cursor="hand2", state=tk.DISABLED)
    detect_btn.pack(pady=5)

    # ------------------------------------------------------------------------
    # RIGHT SIDE: THRESHOLD SLIDER (single class)
    # ------------------------------------------------------------------------
    threshold_frame = tk.LabelFrame(bottom_container, text="⚙️ Detection Threshold", font=("Arial", 14, "bold"), bg=root_bg, fg="#1976D2", padx=20, pady=10)
    threshold_frame.pack(side=tk.RIGHT, padx=20, pady=(0, 40))

    # Create slider for the single "Defect" class
    threshold_vars = {}
    threshold_labels = {}

    for idx, obj_class in enumerate(Object_classes):
        # Control frame for the class
        control_frame = tk.Frame(threshold_frame, bg="white")
        control_frame.pack(pady=5, fill=tk.X)

        # Class name label
        class_label = tk.Label(control_frame, text=f"{obj_class.upper()}:", font=("Arial", 12, "bold"), bg="white", width=10, anchor="w")
        class_label.pack(side=tk.LEFT, padx=10)

        # Threshold variable
        threshold_vars[obj_class] = tk.DoubleVar(value=class_conf[obj_class])

        # Update function for slider
        def make_update_function(obj_class):
            def update_threshold(val):
                class_conf[obj_class] = float(val)
                threshold_labels[obj_class].config(text=f"{float(val):.2f}")
                print(f"Updated confidence for {obj_class}: {val}")
            return update_threshold

        # Slider
        slider = tk.Scale(control_frame, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, variable=threshold_vars[obj_class], command=make_update_function(obj_class), bg="white", highlightthickness=0, length=250, showvalue=0)
        slider.pack(side=tk.LEFT, padx=10)

        # Current value label
        threshold_labels[obj_class] = tk.Label(control_frame, text=f"{class_conf[obj_class]:.2f}", font=("Arial", 14, "bold"), bg="white", fg="#2e7d32", width=6)
        threshold_labels[obj_class].pack(side=tk.LEFT, padx=10)

    # Start the application
    root.mainloop()
