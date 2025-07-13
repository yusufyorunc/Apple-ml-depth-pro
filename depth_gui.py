import os
import traceback
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum

try:
    import depth_pro

    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False


class ColorMap(Enum):
    INFERNO = cv2.COLORMAP_INFERNO
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    PLASMA = cv2.COLORMAP_PLASMA
    MAGMA = cv2.COLORMAP_MAGMA
    JET = cv2.COLORMAP_JET
    COOL = cv2.COLORMAP_COOL
    HOT = cv2.COLORMAP_HOT
    RAINBOW = cv2.COLORMAP_RAINBOW
    PARULA = cv2.COLORMAP_PARULA
    TURBO = cv2.COLORMAP_TURBO


@dataclass
class ProcessingConfig:
    input_path: str
    output_path: str
    colormap: ColorMap
    save_raw: bool = False
    save_colored: bool = True
    normalize_depth: bool = True
    invert_depth: bool = False
    contrast_factor: float = 1.0
    brightness_factor: float = 0.0


class DepthProGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Pro - Professional Depth Estimation")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 600)
        self.setup_theme()
        self.model = None
        self.transform = None
        self.current_image = None
        self.current_depth = None
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.settings_file = Path("depth_pro_settings.json")
        self.create_widgets()
        self.load_settings()
        self.load_model_async()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_theme(self):
        style = ttk.Style()
        style.theme_use("clam")
        self.colors = {
            "bg": "#2b2b2b",
            "fg": "#ffffff",
            "select_bg": "#404040",
            "select_fg": "#ffffff",
            "accent": "#0078d4",
            "success": "#107c10",
            "warning": "#ffb900",
            "error": "#d13438",
            "border": "#454545",
        }

        style.configure(
            "TLabel", background=self.colors["bg"], foreground=self.colors["fg"]
        )
        style.configure(
            "TButton", background=self.colors["select_bg"], foreground=self.colors["fg"]
        )
        style.configure("TFrame", background=self.colors["bg"])
        style.configure(
            "TLabelFrame", background=self.colors["bg"], foreground=self.colors["fg"]
        )
        style.configure("TNotebook", background=self.colors["bg"])
        style.configure(
            "TNotebook.Tab",
            background=self.colors["select_bg"],
            foreground=self.colors["fg"],
        )
        style.configure("TProgressbar", background=self.colors["accent"])

        self.root.configure(bg=self.colors["bg"])

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        self.create_header(main_frame)
        self.create_main_content(main_frame)
        self.create_footer(main_frame)

    def create_header(self, parent):
        header_frame = ttk.Frame(parent)
        header_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        header_frame.columnconfigure(1, weight=1)
        title_label = ttk.Label(
            header_frame, text="Depth Pro", font=("Arial", 20, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        self.status_frame = ttk.Frame(header_frame)
        self.status_frame.grid(row=0, column=1, sticky=tk.E)
        self.status_label = ttk.Label(
            self.status_frame, text="Loading...", font=("Arial", 10)
        )
        self.status_label.grid(row=0, column=0, padx=(0, 5))
        self.model_status = tk.Canvas(
            self.status_frame,
            width=12,
            height=12,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        self.model_status.grid(row=0, column=1)
        self.model_status.create_oval(
            2, 2, 10, 10, fill=self.colors["warning"], outline=""
        )

    def create_main_content(self, parent):
        left_panel = ttk.LabelFrame(parent, text="Controls", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        right_panel = ttk.Frame(parent)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_controls(left_panel)
        self.create_image_display(right_panel)

    def create_controls(self, parent):
        current_row = 0
        file_frame = ttk.LabelFrame(parent, text="File Selection", padding="5")
        file_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        current_row += 1
        ttk.Label(file_frame, text="Input Image:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.input_path_var = tk.StringVar()
        input_entry = ttk.Entry(
            file_frame, textvariable=self.input_path_var, state="readonly"
        )
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)

        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(
            row=0, column=2, padx=(5, 0), pady=2
        )

        ttk.Label(file_frame, text="Output Directory:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(
            file_frame, textvariable=self.output_path_var, state="readonly"
        )
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)

        ttk.Button(file_frame, text="Browse", command=self.browse_output_dir).grid(
            row=1, column=2, padx=(5, 0), pady=2
        )
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding="5")
        options_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        current_row += 1
        ttk.Label(options_frame, text="Color Map:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.colormap_var = tk.StringVar(value="INFERNO")
        colormap_combo = ttk.Combobox(
            options_frame,
            textvariable=self.colormap_var,
            values=[cm.name for cm in ColorMap],
            state="readonly",
        )
        colormap_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        colormap_combo.bind("<<ComboboxSelected>>", self.on_colormap_change)
        self.save_raw_var = tk.BooleanVar(value=False)
        self.save_colored_var = tk.BooleanVar(value=True)
        self.normalize_var = tk.BooleanVar(value=True)
        self.invert_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(
            options_frame, text="Save Raw Depth", variable=self.save_raw_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(
            options_frame, text="Save Colored Depth", variable=self.save_colored_var
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(
            options_frame, text="Normalize Depth", variable=self.normalize_var
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(
            options_frame, text="Invert Depth", variable=self.invert_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)

        adj_frame = ttk.LabelFrame(parent, text="Image Adjustments", padding="5")
        adj_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        adj_frame.columnconfigure(1, weight=1)
        current_row += 1

        ttk.Label(adj_frame, text="Contrast:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(
            adj_frame,
            from_=0.1,
            to=3.0,
            variable=self.contrast_var,
            orient=tk.HORIZONTAL,
        )
        contrast_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        contrast_scale.bind("<Motion>", self.on_adjustment_change)

        ttk.Label(adj_frame, text="Brightness:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        self.brightness_var = tk.DoubleVar(value=0.0)
        brightness_scale = ttk.Scale(
            adj_frame,
            from_=-50,
            to=50,
            variable=self.brightness_var,
            orient=tk.HORIZONTAL,
        )
        brightness_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        brightness_scale.bind("<Motion>", self.on_adjustment_change)

        button_frame = ttk.Frame(parent)
        button_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        current_row += 1

        self.process_button = ttk.Button(
            button_frame, text="Process Image", command=self.process_image
        )
        self.process_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        self.batch_button = ttk.Button(
            button_frame, text="Batch Process", command=self.batch_process
        )
        self.batch_button.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

        quick_frame = ttk.LabelFrame(parent, text="Quick Actions", padding="5")
        quick_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        current_row += 1

        ttk.Button(
            quick_frame, text="Reset Settings", command=self.reset_settings
        ).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(quick_frame, text="Save Settings", command=self.save_settings).grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=2
        )
        ttk.Button(quick_frame, text="Load Settings", command=self.load_settings).grid(
            row=2, column=0, sticky=(tk.W, tk.E), pady=2
        )

        info_frame = ttk.LabelFrame(parent, text="Information", padding="5")
        info_frame.grid(row=current_row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_frame.rowconfigure(0, weight=1)

        self.info_text = tk.Text(
            info_frame,
            height=6,
            width=30,
            wrap=tk.WORD,
            bg=self.colors["select_bg"],
            fg=self.colors["fg"],
            font=("Consolas", 9),
        )
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        info_scroll = ttk.Scrollbar(
            info_frame, orient=tk.VERTICAL, command=self.info_text.yview
        )
        info_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scroll.set)

    def create_image_display(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        original_frame = ttk.Frame(notebook)
        notebook.add(original_frame, text="Original Image")

        self.original_canvas = tk.Canvas(original_frame, bg=self.colors["select_bg"])
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        depth_frame = ttk.Frame(notebook)
        notebook.add(depth_frame, text="Depth Map")

        self.depth_canvas = tk.Canvas(depth_frame, bg=self.colors["select_bg"])
        self.depth_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        depth_frame.columnconfigure(0, weight=1)
        depth_frame.rowconfigure(0, weight=1)

        self.add_scrollbars(original_frame, self.original_canvas)
        self.add_scrollbars(depth_frame, self.depth_canvas)

    def add_scrollbars(self, parent, canvas):
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        canvas.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        canvas.configure(xscrollcommand=h_scrollbar.set)

    def create_footer(self, parent):
        footer_frame = ttk.Frame(parent)
        footer_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0)
        )
        footer_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer_frame, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.footer_status = ttk.Label(footer_frame, text="Ready")
        self.footer_status.grid(row=0, column=1, sticky=tk.E)

    def load_model_async(self):
        def load_model():
            try:
                if not DEPTH_PRO_AVAILABLE:
                    raise ImportError("Depth Pro module not available")

                self.log_info("Loading Depth Pro model...")
                self.model, self.transform = depth_pro.create_model_and_transforms()
                self.model.eval()

                self.root.after(0, self.on_model_loaded)
                self.log_info("Model loaded successfully!")

            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))

        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()

    def on_model_loaded(self):
        self.status_label.config(text="Model Ready")
        self.model_status.create_oval(
            2, 2, 10, 10, fill=self.colors["success"], outline=""
        )
        self.process_button.config(state="normal")
        self.batch_button.config(state="normal")

    def on_model_error(self, error_msg):
        self.status_label.config(text="Model Error")
        self.model_status.create_oval(
            2, 2, 10, 10, fill=self.colors["error"], outline=""
        )
        self.log_error(f"Model loading failed: {error_msg}")
        messagebox.showerror(
            "Model Error", f"Failed to load Depth Pro model:\n{error_msg}"
        )

    def browse_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.input_path_var.set(file_path)
            self.load_input_image(file_path)

    def browse_output_dir(self):
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_path_var.set(dir_path)

    def load_input_image(self, file_path):
        try:
            self.current_image = cv2.imread(file_path)
            if self.current_image is None:
                raise ValueError("Could not load image")
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            self.display_image(self.original_canvas, pil_image)
            height, width = self.current_image.shape[:2]
            file_size = os.path.getsize(file_path) / 1024 / 1024
            self.log_info(f"Loaded: {os.path.basename(file_path)}")
            self.log_info(f"Size: {width}x{height}")
            self.log_info(f"File size: {file_size:.2f} MB")

        except Exception as e:
            self.log_error(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

    def display_image(self, canvas, pil_image):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, lambda: self.display_image(canvas, pil_image))
            return

        img_width, img_height = pil_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        scale = min(scale_x, scale_y, 1.0)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        display_image = pil_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )
        photo = ImageTk.PhotoImage(display_image)
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo
        )
        canvas.image = photo
        canvas.configure(scrollregion=canvas.bbox("all"))

    def process_image(self):
        if not self.validate_inputs():
            return

        if self.is_processing:
            messagebox.showwarning("Processing", "Another process is already running.")
            return

        config = self.get_processing_config()
        thread = threading.Thread(
            target=self.process_image_thread, args=(config,), daemon=True
        )
        thread.start()

    def process_image_thread(self, config):
        try:
            self.is_processing = True
            self.root.after(0, self.start_processing_ui)
            self.root.after(0, lambda: self.log_info("Processing image..."))
            image, _, f_px = depth_pro.load_rgb(config.input_path)
            image = self.transform(image)
            self.root.after(0, lambda: self.log_info("Running inference..."))
            prediction = self.model.infer(image, f_px=f_px)
            depth = prediction["depth"]
            depth_np = depth.squeeze().cpu().numpy()
            self.current_depth = depth_np.copy()
            if config.invert_depth:
                depth_np = np.max(depth_np) - depth_np

            if config.normalize_depth:
                depth_np = (depth_np - depth_np.min()) / (
                    depth_np.max() - depth_np.min()
                )
            depth_np = (
                depth_np * config.contrast_factor + config.brightness_factor / 100.0
            )
            depth_np = np.clip(depth_np, 0, 1)
            depth_uint8 = (depth_np * 255).astype(np.uint8)
            base_name = Path(config.input_path).stem
            output_dir = Path(config.output_path)

            if config.save_raw:
                raw_path = output_dir / f"{base_name}_depth_raw.png"
                cv2.imwrite(str(raw_path), depth_uint8)
                self.root.after(
                    0, lambda: self.log_info(f"Saved raw depth: {raw_path.name}")
                )

            if config.save_colored:
                colored_depth = cv2.applyColorMap(depth_uint8, config.colormap.value)
                colored_path = output_dir / f"{base_name}_depth_colored.png"
                cv2.imwrite(str(colored_path), colored_depth)
                self.root.after(
                    0,
                    lambda: self.log_info(f"Saved colored depth: {colored_path.name}"),
                )
                colored_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
                pil_colored = Image.fromarray(colored_rgb)
                self.root.after(
                    0, lambda: self.display_image(self.depth_canvas, pil_colored)
                )

            self.root.after(
                0, lambda: self.log_info("Processing completed successfully!")
            )

        except Exception as e:
            self.root.after(0, lambda: self.log_error(f"Processing error: {str(e)}"))
            self.root.after(
                0,
                lambda: messagebox.showerror("Error", f"Processing failed:\n{str(e)}"),
            )

        finally:
            self.is_processing = False
            self.root.after(0, self.stop_processing_ui)

    def batch_process(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded")
            return

        input_dir = filedialog.askdirectory(title="Select Input Directory")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        input_path = Path(input_dir)
        image_files = [
            f for f in input_path.iterdir() if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            messagebox.showwarning(
                "Warning", "No image files found in selected directory"
            )
            return
        result = messagebox.askyesno(
            "Batch Processing", f"Process {len(image_files)} images?"
        )
        if not result:
            return

        thread = threading.Thread(
            target=self.batch_process_thread,
            args=(image_files, output_dir),
            daemon=True,
        )
        thread.start()

    def batch_process_thread(self, image_files, output_dir):
        try:
            self.is_processing = True
            self.root.after(0, self.start_processing_ui)

            total_files = len(image_files)
            processed = 0

            for image_file in image_files:
                try:
                    config = ProcessingConfig(
                        input_path=str(image_file),
                        output_path=output_dir,
                        colormap=ColorMap[self.colormap_var.get()],
                        save_raw=self.save_raw_var.get(),
                        save_colored=self.save_colored_var.get(),
                        normalize_depth=self.normalize_var.get(),
                        invert_depth=self.invert_var.get(),
                        contrast_factor=self.contrast_var.get(),
                        brightness_factor=self.brightness_var.get(),
                    )

                    self.root.after(
                        0, lambda f=image_file: self.log_info(f"Processing: {f.name}")
                    )
                    image, _, f_px = depth_pro.load_rgb(str(image_file))
                    image = self.transform(image)

                    prediction = self.model.infer(image, f_px=f_px)
                    depth = prediction["depth"]

                    depth_np = depth.squeeze().cpu().numpy()

                    if config.invert_depth:
                        depth_np = np.max(depth_np) - depth_np

                    if config.normalize_depth:
                        depth_np = (depth_np - depth_np.min()) / (
                            depth_np.max() - depth_np.min()
                        )

                    depth_np = (
                        depth_np * config.contrast_factor
                        + config.brightness_factor / 100.0
                    )
                    depth_np = np.clip(depth_np, 0, 1)
                    depth_uint8 = (depth_np * 255).astype(np.uint8)

                    base_name = image_file.stem
                    output_path = Path(output_dir)

                    if config.save_raw:
                        raw_path = output_path / f"{base_name}_depth_raw.png"
                        cv2.imwrite(str(raw_path), depth_uint8)

                    if config.save_colored:
                        colored_depth = cv2.applyColorMap(
                            depth_uint8, config.colormap.value
                        )
                        colored_path = output_path / f"{base_name}_depth_colored.png"
                        cv2.imwrite(str(colored_path), colored_depth)

                    processed += 1
                    progress = (processed / total_files) * 100
                    self.root.after(0, lambda p=progress: self.update_batch_progress(p))

                except Exception as e:
                    self.root.after(
                        0,
                        lambda f=image_file, err=str(e): self.log_error(
                            f"Error processing {f.name}: {err}"
                        ),
                    )

            self.root.after(
                0,
                lambda: self.log_info(
                    f"Batch processing completed! Processed {processed}/{total_files} images."
                ),
            )

        except Exception as e:
            self.root.after(
                0, lambda: self.log_error(f"Batch processing error: {str(e)}")
            )

        finally:
            self.is_processing = False
            self.root.after(0, self.stop_processing_ui)

    def update_batch_progress(self, progress):
        self.footer_status.config(text=f"Processing: {progress:.1f}%")

    def validate_inputs(self):
        if not self.input_path_var.get():
            messagebox.showerror("Error", "Please select an input image")
            return False

        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an output directory")
            return False

        if not self.model:
            messagebox.showerror("Error", "Model not loaded")
            return False

        return True

    def get_processing_config(self):
        return ProcessingConfig(
            input_path=self.input_path_var.get(),
            output_path=self.output_path_var.get(),
            colormap=ColorMap[self.colormap_var.get()],
            save_raw=self.save_raw_var.get(),
            save_colored=self.save_colored_var.get(),
            normalize_depth=self.normalize_var.get(),
            invert_depth=self.invert_var.get(),
            contrast_factor=self.contrast_var.get(),
            brightness_factor=self.brightness_var.get(),
        )

    def start_processing_ui(self):
        self.progress.start()
        self.process_button.config(state="disabled")
        self.batch_button.config(state="disabled")
        self.footer_status.config(text="Processing...")

    def stop_processing_ui(self):
        self.progress.stop()
        self.process_button.config(state="normal")
        self.batch_button.config(state="normal")
        self.footer_status.config(text="Ready")

    def on_colormap_change(self, event=None):
        if self.current_depth is not None:
            self.update_depth_preview()

    def on_adjustment_change(self, event=None):
        if self.current_depth is not None:
            if hasattr(self, "_update_timer"):
                self.root.after_cancel(self._update_timer)
            self._update_timer = self.root.after(100, self.update_depth_preview)

    def update_depth_preview(self):
        if self.current_depth is None:
            return

        try:
            depth_np = self.current_depth.copy()

            if self.invert_var.get():
                depth_np = np.max(depth_np) - depth_np

            if self.normalize_var.get():
                depth_np = (depth_np - depth_np.min()) / (
                    depth_np.max() - depth_np.min()
                )

            depth_np = (
                depth_np * self.contrast_var.get() + self.brightness_var.get() / 100.0
            )
            depth_np = np.clip(depth_np, 0, 1)

            depth_uint8 = (depth_np * 255).astype(np.uint8)
            colormap = ColorMap[self.colormap_var.get()]
            colored_depth = cv2.applyColorMap(depth_uint8, colormap.value)

            colored_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
            pil_colored = Image.fromarray(colored_rgb)
            self.display_image(self.depth_canvas, pil_colored)

        except Exception as e:
            self.log_error(f"Preview update error: {str(e)}")

    def log_info(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)

    def log_error(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] ERROR: {message}\n")
        self.info_text.see(tk.END)

    def save_settings(self):
        try:
            if not hasattr(self, "colormap_var"):
                return

            settings = {
                "colormap": self.colormap_var.get(),
                "save_raw": self.save_raw_var.get(),
                "save_colored": self.save_colored_var.get(),
                "normalize_depth": self.normalize_var.get(),
                "invert_depth": self.invert_var.get(),
                "contrast_factor": self.contrast_var.get(),
                "brightness_factor": self.brightness_var.get(),
                "output_path": self.output_path_var.get(),
            }

            with open(self.settings_file, "w") as f:
                json.dump(settings, f, indent=2)

            if hasattr(self, "info_text"):
                self.log_info("Settings saved successfully")
                messagebox.showinfo("Success", "Settings saved successfully")

        except Exception as e:
            if hasattr(self, "info_text"):
                self.log_error(f"Error saving settings: {str(e)}")
                messagebox.showerror("Error", f"Failed to save settings:\n{str(e)}")
            else:
                print(f"Error saving settings: {str(e)}")

    def load_settings(self):
        try:
            if self.settings_file.exists():
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                if hasattr(self, "colormap_var"):
                    self.colormap_var.set(settings.get("colormap", "INFERNO"))
                if hasattr(self, "save_raw_var"):
                    self.save_raw_var.set(settings.get("save_raw", False))
                if hasattr(self, "save_colored_var"):
                    self.save_colored_var.set(settings.get("save_colored", True))
                if hasattr(self, "normalize_var"):
                    self.normalize_var.set(settings.get("normalize_depth", True))
                if hasattr(self, "invert_var"):
                    self.invert_var.set(settings.get("invert_depth", False))
                if hasattr(self, "contrast_var"):
                    self.contrast_var.set(settings.get("contrast_factor", 1.0))
                if hasattr(self, "brightness_var"):
                    self.brightness_var.set(settings.get("brightness_factor", 0.0))
                if hasattr(self, "output_path_var"):
                    self.output_path_var.set(settings.get("output_path", ""))

                if hasattr(self, "info_text"):
                    self.log_info("Settings loaded successfully")

        except Exception as e:
            if hasattr(self, "info_text"):
                self.log_error(f"Error loading settings: {str(e)}")
            else:
                print(f"Error loading settings: {str(e)}")

    def reset_settings(self):
        self.colormap_var.set("INFERNO")
        self.save_raw_var.set(False)
        self.save_colored_var.set(True)
        self.normalize_var.set(True)
        self.invert_var.set(False)
        self.contrast_var.set(1.0)
        self.brightness_var.set(0.0)

        self.log_info("Settings reset to default")

    def on_closing(self):
        if hasattr(self, "is_processing") and self.is_processing:
            result = messagebox.askyesno(
                "Confirm Exit",
                "Processing is in progress. Are you sure you want to exit?",
            )
            if not result:
                return
        try:
            self.save_settings()
        except Exception as e:
            print(f"Error saving settings on exit: {str(e)}")
        try:
            if hasattr(self, "progress"):
                self.progress.stop()
        except:
            pass

        self.root.destroy()


class DepthProSplashScreen:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Depth Pro")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (300 // 2)
        self.root.geometry(f"400x300+{x}+{y}")
        self.root.overrideredirect(True)
        self.create_splash()
        self.root.after(3000, self.close_splash)

    def create_splash(self):
        main_frame = tk.Frame(
            self.root, bg="#1a1a1a", highlightbackground="#0078d4", highlightthickness=2
        )
        main_frame.pack(fill=tk.BOTH, expand=True)
        title_label = tk.Label(
            main_frame,
            text="Depth Pro",
            font=("Arial", 24, "bold"),
            bg="#1a1a1a",
            fg="#ffffff",
        )
        title_label.pack(pady=(50, 10))
        subtitle_label = tk.Label(
            main_frame,
            text="Professional Depth Estimation Tool",
            font=("Arial", 12),
            bg="#1a1a1a",
            fg="#cccccc",
        )
        subtitle_label.pack(pady=(0, 30))
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate", length=250)
        self.progress.pack(pady=20)
        self.progress.start()
        self.status_label = tk.Label(
            main_frame,
            text="Loading...",
            font=("Arial", 10),
            bg="#1a1a1a",
            fg="#999999",
        )
        self.status_label.pack(pady=(10, 0))
        version_label = tk.Label(
            main_frame, text="v1.0.0", font=("Arial", 8), bg="#1a1a1a", fg="#666666"
        )
        version_label.pack(side=tk.BOTTOM, pady=10)

    def close_splash(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    try:
        splash = DepthProSplashScreen()
        splash.run()
        root = tk.Tk()
        app = DepthProGUI(root)
        root.mainloop()

    except Exception as e:
        print(f"Application error: {str(e)}")
        traceback.print_exc()
        messagebox.showerror(
            "Critical Error", f"Application failed to start:\n{str(e)}"
        )


if __name__ == "__main__":
    main()
