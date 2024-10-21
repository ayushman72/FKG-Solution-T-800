from pathlib import Path
from tkinter import Tk, Canvas, Label, VERTICAL, Scrollbar, Text, Button, PhotoImage, filedialog 
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO        #type: ignore
from paddleocr import PaddleOCR     #type: ignore
from collections import Counter

model = YOLO("fkg_model.pt")
reader = PaddleOCR(use_angle_cls=True, lang='en',show_log=False)
ASSETS_PATH = Path(r"assets")

global k
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def create_rounded_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1 + radius, y1,
              x1 + radius, y1,
              x2 - radius, y1,
              x2 - radius, y1,
              x2, y1,
              x2, y1 + radius,
              x2, y1 + radius,
              x2, y2 - radius,
              x2, y2 - radius,
              x2, y2,
              x2 - radius, y2,
              x2 - radius, y2,
              x1 + radius, y2,
              x1 + radius, y2,
              x1, y2,
              x1, y2 - radius,
              x1, y2 - radius,
              x1, y1 + radius,
              x1, y1 + radius,
              x1, y1]

    return canvas.create_polygon(points, smooth=True, **kwargs)

def stop_live_camera():
    global capturing
    capturing = False
    cap.release()

def load_image():
    
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;.png;")])
    if image_path:  
        img = Image.open(image_path)
        img = img.resize((667, 679))
        img = ImageTk.PhotoImage(img)
        canvas.create_image(33, 88, anchor="nw", image=img)
        canvas.image = img

def predict_image():
    frame = cv2.imread(image_path)
    ocr = reader.ocr(frame)
    try:
        out = " ".join([i[1][0] for i in ocr[0]])
    except:
        out = " "
        
    frame = cv2.resize(frame, (480, 640))
    results = model.predict(frame,iou=0.3,verbose=False)
    box = results[0].boxes

    for b,l in zip(box.xyxy, box.cls):
        label = results[0].names[l.item()]
        boundary = b.int().tolist()
        frame = cv2.rectangle(frame, boundary[:2], boundary[2:], (0, 255, 0), 2)
        frame = cv2.putText(frame, label, boundary[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    img_tk = ImageTk.PhotoImage(img_pil.resize((667, 679)))

    canvas.create_image(37, 88, anchor="nw", image=img_tk)
    canvas.image = img_tk

    products = Counter(box.cls.int().tolist())
    text = ""
    for product, count in products.items():
        text += f"{results[0].names[product]}: {count}\n"
    brand_text.config(state="normal")
    brand_text.delete("1.0", "end")
    brand_text.insert("end", out)
    brand_details.config(state="normal")
    brand_details.delete("1.0", "end")
    brand_details.insert("end", text)


def start_live_camera():
    global capturing, cap, last_frames,buffer
    capturing = True
    cap = cv2.VideoCapture(1)  # Use the appropriate index for your camera
    last_frames = [set()]
    buffer = 1
    update_frame()

def update_frame():
    
    if capturing:
        ret, frame = cap.read()
        ocr = reader.ocr(frame)
        try:
            out = " ".join([i[1][0] for i in ocr[0]])
        except:
            out = " "
        results = model.predict(frame,iou=0.3,verbose=False)
        box = results[0].boxes
        
        persist = last_frames[0].intersection(*last_frames)
        
        for b,l in zip(box.xyxy, box.cls):
            if l.item() not in persist:
                continue
            label = results[0].names[l.item()]
            boundary = b.int().tolist()
            frame = cv2.rectangle(frame, boundary[:2], boundary[2:], (0, 255, 0), 2)
            frame = cv2.putText(frame, label, boundary[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                
                
        if len(last_frames) > buffer:
            last_frames.pop(0)
        last_frames.append(set(box.cls.tolist()))

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        img_tk = ImageTk.PhotoImage(img_pil.resize((667, 679)))

        canvas.create_image(37, 88, anchor="nw", image=img_tk)
        canvas.image = img_tk

        products = Counter([i for i in box.cls.int().tolist() if i in persist])
        text = ""
        for product, count in products.items():
            text += f"{results[0].names[product]}: {count}\n"

        brand_text.config(state="normal")
        brand_details.config(state="normal")
        brand_text.delete(1.0, "end")
        brand_details.delete("1.0", "end")
        brand_text.insert("end", out)
        brand_details.insert("end", text)
        brand_text.config(state="disabled")
        brand_details.config(state="disabled")   
    
        window.after(10, update_frame)

window = Tk()
window.geometry("1600x850")
window.configure(bg = "#041813")

canvas = Canvas(
    window,
    bg = "#041813",
    height = 850,
    width = 1600,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    1600.0,
    850.0,
    fill="#041813",
    outline="")

scrollbar_text = Scrollbar(window, orient=VERTICAL)
scrollbar_details = Scrollbar(window, orient=VERTICAL)
brand_text = Text(window, wrap='word', yscrollcommand=scrollbar_details.set, bg="#3B3B3B", fg="white", font=("InriaSans Regular", 20), height=7, width=30)
brand_details = Text(window, wrap='word', yscrollcommand=scrollbar_text.set, bg="#3B3B3B", fg="white", font=("InriaSans Regular", 20), height=7, width=30)
scrollbar_text.config(command=brand_details.yview)
scrollbar_details.config(command=brand_text.yview)

scrollbar_text.place(x=1478, y=117, height=227)
brand_details.place(x=1026, y=117)

scrollbar_details.place(x=1478, y=457, height=227)
brand_text.place(x=1026, y=457)

button_image = PhotoImage(
    file=relative_to_assets("button_2.png"))
button = Button(
    image=button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button.place(
    x=1172.9345703125,
    y=700.2258605957031,
    width=266.1308288574219,
    height=50.331512451171875
)

canvas.create_rectangle(
    37.0,
    88.0,
    704.0,
    767.0,
    fill="#3B3B3B",
    outline="")

button_image_2 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_2 clicked"),
    relief="flat"
)
button_2.place(
    x=1172.9345703125,
    y=365.2258605957031,
    width=266.1308288574219,
    height=50.331512451171875
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=start_live_camera,
    relief="flat"
)
button_3.place(
    x=756.0,
    y=479.5015869140625,
    width=218.0,
    height=53.30519104003906
)

button_image_hover_3 = PhotoImage(
    file=relative_to_assets("button_hover_1.png"))

def button_3_hover(e):
    button_3.config(
        image=button_image_hover_3
    )
def button_3_leave(e):
    button_3.config(
        image=button_image_3
    )

button_3.bind('<Enter>', button_3_hover)
button_3.bind('<Leave>', button_3_leave)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=predict_image,
    relief="flat"
)
button_4.place(
    x=756.0,
    y=309.0,
    width=218.0,
    height=41.02587890625
)

button_image_hover_4 = PhotoImage(
    file=relative_to_assets("button_hover_2.png"))

def button_4_hover(e):
    button_4.config(
        image=button_image_hover_4
    )
def button_4_leave(e):
    button_4.config(
        image=button_image_4
    )

button_4.bind('<Enter>', button_4_hover)
button_4.bind('<Leave>', button_4_leave)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=load_image,
    relief="flat"
)
button_5.place(
    x=250.0,
    y=781.0,
    width=241.0,
    height=50.0
)

button_image_hover_5 = PhotoImage(
    file=relative_to_assets("button_hover_3.png"))

def button_5_hover(e):
    button_5.config(
        image=button_image_hover_5
    )
def button_5_leave(e):
    button_5.config(
        image=button_image_5
    )

button_5.bind('<Enter>', button_5_hover)
button_5.bind('<Leave>', button_5_leave)

stop_image_button = PhotoImage(
    file=relative_to_assets("file.png"))
button_stop_camera = Button(
    image=stop_image_button,  
    borderwidth=0,
    highlightthickness=0,
    command=stop_live_camera,
    relief="flat"
)
button_stop_camera.place(
    x=817,  
    y=625,
    width=95,
    height=60
)

canvas.create_rectangle(
    0.0,
    0.0,
    1600.0,
    71.0,
    fill="#002305",
    outline="")

canvas.create_text(
    749.0,
    6.0,
    anchor="nw",
    text="T-800",
    fill="#FEDC12",
    font=("InriaSans Regular", 55 * -1)
)

canvas.create_text(
    0.0,
    10.0,
    anchor="nw",
    text="Flipkart GRiD",
    fill="#FEDC12",
    font=("InriaSans Regular", 38 * -1)
)
count_label = Label(window, text="", bg="#3B3B3B", fg="white", font=("InriaSans Regular", 20))
count_label.place(x=1100, y=119)
window.resizable(True, True)
window.mainloop()