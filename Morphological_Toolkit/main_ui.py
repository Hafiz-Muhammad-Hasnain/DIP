import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np

from core import BinaryImage, StructuringElement, MorphOp, ShapeAnalyzer, BatchProcessor, AIEngine

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MorphToolkitUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Morphological Image Processing Toolkit (PRO)")
        self.geometry("1400x900")
        self.minsize(1000, 700)
        
        self.bin_img = BinaryImage()
        self.processed_img = None
        self.analyzed_img = None
        
        self.setup_ui()
        
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar_frame, text="Morphology Studio", font=ctk.CTkFont(size=20, weight="bold")).pack(padx=20, pady=(20, 10))
        
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(padx=20, pady=10)
        
        self.bin_button = ctk.CTkButton(self.sidebar_frame, text="Binarize (Otsu)", command=self.binarize_image)
        self.bin_button.pack(padx=20, pady=10)
        
        self.ai_bin_button = ctk.CTkButton(self.sidebar_frame, text="AI Binarize (DeepLab/U-Net)", command=self.ai_binarize_image, fg_color="#9933ff", hover_color="#7700cc")
        self.ai_bin_button.pack(padx=20, pady=10)
        
        # Kernel Area
        ctk.CTkLabel(self.sidebar_frame, text="Structuring Element:", anchor="w").pack(padx=20, pady=(10, 0), fill="x")
        self.kernel_shape = ctk.StringVar(value="rect")
        ctk.CTkOptionMenu(self.sidebar_frame, values=["rect", "cross", "ellipse", "custom"], variable=self.kernel_shape, command=self.toggle_custom_kernel).pack(padx=20, pady=5, fill="x")
        
        self.custom_kernel_frame = ctk.CTkFrame(self.sidebar_frame)
        self.custom_kernel_data = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.custom_kernel_btns = []
        for r in range(3):
            self.custom_kernel_frame.grid_columnconfigure(r, weight=1)
            row_btns = []
            for c in range(3):
                btn = ctk.CTkButton(self.custom_kernel_frame, text="1", width=30, height=30, 
                                    command=lambda r=r, c=c: self.toggle_kernel_val(r, c))
                btn.grid(row=r, column=c, padx=2, pady=2)
                row_btns.append(btn)
            self.custom_kernel_btns.append(row_btns)

        ctk.CTkLabel(self.sidebar_frame, text="Kernel Size:", anchor="w").pack(padx=20, pady=(10, 0), fill="x")
        self.kernel_size = ctk.IntVar(value=3)
        ctk.CTkSlider(self.sidebar_frame, from_=3, to=15, number_of_steps=6, command=lambda v: self.kernel_size.set(int(v))).pack(padx=20, pady=5, fill="x")
        
        # Operations
        ctk.CTkLabel(self.sidebar_frame, text="Morph Operations:", anchor="w").pack(padx=20, pady=(10, 0), fill="x")
        self.op_var = ctk.StringVar(value="erode")
        ctk.CTkOptionMenu(self.sidebar_frame, values=["erode", "dilate", "open", "close", "hit-or-miss", "gradient"], variable=self.op_var).pack(padx=20, pady=5, fill="x")
        
        ctk.CTkButton(self.sidebar_frame, text="Apply Operation", command=self.apply_morph, fg_color="#28a745", hover_color="#218838").pack(padx=20, pady=15, fill="x")
        
        ctk.CTkButton(self.sidebar_frame, text="Shape Analysis", command=self.analyze_shape).pack(padx=20, pady=10, fill="x")
        self.ai_analyze_btn = ctk.CTkButton(self.sidebar_frame, text="AI Detect & Classify", command=self.ai_analyze_shape, fg_color="#ff33cc", hover_color="#cc0099")
        self.ai_analyze_btn.pack(padx=20, pady=10, fill="x")
        self.shape_info = ctk.CTkLabel(self.sidebar_frame, text="Objects: 0")
        self.shape_info.pack(padx=20, pady=5)
        
        self.save_button = ctk.CTkButton(self.sidebar_frame, text="Save Result", command=self.save_image)
        self.save_button.pack(padx=20, pady=10, fill="x")
        
        # Main Area
        self.main_tabview = ctk.CTkTabview(self)
        self.main_tabview.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.tab_single = self.main_tabview.add("Interactive Mode")
        self.tab_batch = self.main_tabview.add("Batch Processing")
        
        # Interactive Tab
        self.tab_single.grid_columnconfigure((0, 1), weight=1)
        self.tab_single.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.tab_single, text="Original/Binarized", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0)
        ctk.CTkLabel(self.tab_single, text="Result/Analysis", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1)
        
        self.orig_label = ctk.CTkLabel(self.tab_single, text="No Image")
        self.orig_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        self.res_label = ctk.CTkLabel(self.tab_single, text="No Result")
        self.res_label.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        
        # Batch Tab
        self.tab_batch.grid_columnconfigure(0, weight=1)
        self.batch_in_dir = ctk.StringVar()
        self.batch_out_dir = ctk.StringVar()
        
        f = ctk.CTkFrame(self.tab_batch)
        f.pack(pady=40, padx=40, fill="x")
        
        ctk.CTkButton(f, text="Select Input Directory", command=lambda: self.batch_in_dir.set(filedialog.askdirectory())).pack(pady=10)
        ctk.CTkLabel(f, textvariable=self.batch_in_dir).pack(pady=5)
        
        ctk.CTkButton(f, text="Select Output Directory", command=lambda: self.batch_out_dir.set(filedialog.askdirectory())).pack(pady=10)
        ctk.CTkLabel(f, textvariable=self.batch_out_dir).pack(pady=5)
        
        ctk.CTkButton(f, text="Run Batch Processing", command=self.run_batch, fg_color="#ff9900", hover_color="#cc7a00").pack(pady=20)
        
    def toggle_custom_kernel(self, choice):
        if choice == "custom":
            self.custom_kernel_frame.pack(padx=20, pady=5, fill="x") 
        else:
            self.custom_kernel_frame.pack_forget()

    def toggle_kernel_val(self, r, c):
        val = self.custom_kernel_data[r][c]
        new_val = 0 if val == 1 else 1
        self.custom_kernel_data[r][c] = new_val
        self.custom_kernel_btns[r][c].configure(text=str(new_val))

    def display_img(self, cv_img, label):
        if cv_img is None: return
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
        pil_img = Image.fromarray(cv_img)
        
        self.update_idletasks()
        # Ensure uniform distribution taking up equal halves of available space
        max_w = (self.winfo_width() - 360) // 2
        max_h = self.winfo_height() - 200
        if max_w < 100: max_w = 400
        if max_h < 100: max_h = 400
        
        img_w, img_h = pil_img.size
        # scale while preserving aspect ratio
        ratio = min(max_w / img_w, max_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        img_tk = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
        label.configure(image=img_tk, text="")
        label.image = img_tk

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.bin_img.load(path)
            self.display_img(self.bin_img.image, self.orig_label)
            self.processed_img = None
            self.masked_img = None
            self.res_label.configure(image=None, text="No Result")
            self.analyzed_img = None
            self.shape_info.configure(text="Objects: 0")
            
    def binarize_image(self):
        if self.bin_img.image is not None:
            self.bin_img.binarize()
            self.processed_img = self.bin_img.binary
            self.display_img(self.processed_img, self.orig_label)
            self.shape_info.configure(text="Objects: 0")
            self.res_label.configure(image=None, text="No Result")
            self.analyzed_img = None

    def ai_binarize_image(self):
        if self.bin_img.image is not None:
            mask = AIEngine.smart_binarize(self.bin_img.image)
            if isinstance(mask, str):
                messagebox.showerror("AI Error", mask)
            else:
                self.bin_img.binary = mask
                self.processed_img = self.bin_img.binary
                self.display_img(self.processed_img, self.orig_label)
                self.shape_info.configure(text="Objects: 0")
                self.res_label.configure(image=None, text="No Result")
                self.analyzed_img = None
                
    def apply_morph(self):
        if self.bin_img.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        if self.bin_img.binary is None:
            # Auto-binarize if user forgot to click Binarize
            self.binarize_image()
            
        if self.processed_img is None:
            self.processed_img = self.bin_img.binary
            
        ksize = self.kernel_size.get()
        shape = self.kernel_shape.get()
        
        if self.op_var.get() == 'hit-or-miss':
            kernel = StructuringElement.create_hit_or_miss()
        elif shape == 'custom':
            kernel = np.array(self.custom_kernel_data, dtype=np.uint8)
        else:
            kernel = StructuringElement.create(shape, (ksize, ksize))
            
        op = self.op_var.get()
        if op == 'erode': self.processed_img = MorphOp.erode(self.processed_img, kernel)
        elif op == 'dilate': self.processed_img = MorphOp.dilate(self.processed_img, kernel)
        elif op == 'open': self.processed_img = MorphOp.opening(self.processed_img, kernel)
        elif op == 'close': self.processed_img = MorphOp.closing(self.processed_img, kernel)
        elif op == 'hit-or-miss': self.processed_img = MorphOp.hit_or_miss(self.processed_img, kernel)
        elif op == 'gradient': self.processed_img = MorphOp.gradient(self.processed_img, kernel)
        
        self.analyzed_img = None
        
        orig = self.bin_img.image
        if orig is not None and len(self.processed_img.shape) == 2:
            self.masked_img = cv2.bitwise_and(orig, orig, mask=self.processed_img)
            self.display_img(self.masked_img, self.res_label)
        else:
            self.masked_img = None
            self.display_img(self.processed_img, self.res_label)
        
    def analyze_shape(self):
        if self.processed_img is not None:
            self.analyzed_img, count = ShapeAnalyzer.analyze(self.processed_img)
            self.display_img(self.analyzed_img, self.res_label)
            self.shape_info.configure(text=f"Objects: {count}")

    def ai_analyze_shape(self):
        if self.processed_img is not None:
            self.analyzed_img, count = ShapeAnalyzer.analyze(self.processed_img, orig_img=self.bin_img.image, use_ai=True)
            self.display_img(self.analyzed_img, self.res_label)
            self.shape_info.configure(text=f"Objects: {count}")

    def save_image(self):
        img_to_save = self.analyzed_img if self.analyzed_img is not None else getattr(self, 'masked_img', self.processed_img)
        if img_to_save is None:
            img_to_save = self.processed_img
            
        if img_to_save is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, img_to_save)
                
    def run_batch(self):
        in_d = self.batch_in_dir.get()
        out_d = self.batch_out_dir.get()
        if not in_d or not out_d:
            messagebox.showerror("Error", "Select input and output directories.")
            return
            
        ksize = self.kernel_size.get()
        kernel = StructuringElement.create(self.kernel_shape.get(), (ksize, ksize))
        op = self.op_var.get()
        
        try:
            c = BatchProcessor.process(in_d, out_d, [op], kernel)
            messagebox.showinfo("Success", f"Processed {c} images.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = MorphToolkitUI()
    app.mainloop()
