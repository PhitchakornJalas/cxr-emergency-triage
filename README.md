# cxr-emergency-triage
## 🎯 ปัญหาและความสำคัญ

**ภาวะลมรั่วในช่องเยื่อหุ้มปอด (Pneumothorax)** เป็นภาวะฉุกเฉินที่อาจเป็นอันตรายถึงชีวิตหากไม่ได้รับการรักษาอย่างทันท่วงที แต่การคัดกรองจากภาพถ่ายรังสีทรวงอกในปัจจุบันยังเผชิญข้อจำกัดสำคัญ 3 ประการ:

1. **ความล่าช้าในการจัดลำดับคิวตรวจ:** ในแผนกฉุกเฉินที่มีปริมาณภาพเอกซเรย์จำนวนมาก เคสผู้ป่วยวิกฤตมักถูกจัดเรียงตามลำดับก่อน-หลัง ปะปนกับเคสปกติ ทำให้รังสีแพทย์เข้าถึงเคสเร่งด่วนได้ช้า
2. **รอยโรคสังเกตได้ยากและข้อมูลไม่สมดุล:** รอยรั่วขนาดเล็กมักเป็นเพียงเส้นขอบเยื่อหุ้มปอดบางๆ ที่ซ้อนทับกับกระดูกซี่โครง ประกอบกับสัดส่วนผู้ป่วยจริงมีน้อยกว่าคนปกติมาก (Class Imbalance) ทำให้โมเดลทั่วไปมักทำนายพลาดในเคสที่ดูยาก
3. **ขาดการอธิบายตำแหน่งรอยโรค:** โมเดลคัดกรองทั่วไปบอกได้เพียงค่าความน่าจะเป็น แต่ไม่สามารถชี้ตำแหน่งบริเวณที่ปอดรั่วให้แพทย์ตรวจสอบยืนยันได้อย่างรวดเร็ว

**แนวทางแก้ไข:** โครงการนี้จึงพัฒนาสถาปัตยกรรม **Hybrid Segmentation + Deep AUC Maximization (DAM)** เพื่อแยกผู้ป่วยออกจากคนปกติ ควบคู่ไปกับการระบายสีระบุตำแหน่งปอดรั่วบนภาพเอกซเรย์

## 📁 โครงสร้างโฟลเดอร์
```text
cxr-emergency-triage/
├── src/                        # โค้ดหลักของระบบ
│   ├── dowload_data.py         # โหลดข้อมูลจาก kaggle SIIM-ACR Pneumothorax Training Data PNG
│   ├── build_registry.py       # สร้างดัชนีไฟล์ภาพ Mask และ Label พร้อมแบ่งชุดข้อมูล
│   ├── split_data.py           # แบ้งข้อมูลอกกเป็น train / val / test
│   ├── show_class_distribution.py # แสดงสัดส่วนของข้อมูลหลังแบ่ง
│   ├── visualize_splits.py     # แสดงภาพตัวอย่างหลังทำการแบ่งข้อมูล
│   ├── preprocess_data.py      # resize เพื่อช่วยลดภาระตอนเทรนโมเดล
│   ├── dataset.py              # โหลดภาพ CXR + RLE Mask, ทำ CLAHE และ Data Augmentation แบบซิงโครไนซ์ 1:1
│   ├── model.py                # สถาปัตยกรรม HybridCXRSegmentationDAM (Shared Encoder + Seg Decoder + DAM Head)
│   ├── train.py                # สคริปต์เทรน 2 ระยะ (Stage 1: Multi-Task Learning / Stage 2: Deep AUC Maximization)
│   ├── plot_history.py         # สร้างกราฟการเรียนรรู้ของโมเดล
│   ├── evaluate.py             # ปรับจุดตัดทางคลินิก (Sensitivity ≥ 95%), คำนวณ Metrics และสร้างกราฟ ROC/PR/CM
│   ├── inference_viz.py        # สร้างภาพเปรียบเทียบผลทำนายแยกกลุ่ม TP / FN / FP / TN เพื่อวิเคราะห์จุดผิดพลาด
│   ├── export_onnx.py          # แปลงโมเดล PyTorch (.pth) เป็น ONNX และสร้างไฟล์ config.json สำหรับหน้าเว็บ
│   └── prepare_demo_images.py  # โหลดภาพที่เป็น test set มาสำหรับลองใช้ทดสอบใน prototype
│
├── data/                       # ชุดข้อมูลภาพเอกซเรย์ปอด
│   ├── registry.csv
│   ├── train.csv              
│   ├── val.csv                 
│   └── test.csv                
│
├── checkpoints/                # เก็บไฟล์น้ำหนักโมเดลที่ดีที่สุด (*_best.pth) 
├── outputs/                    # เก็บผลลัพธ์การประเมิน (*_metrics.json), log_history กราฟ และภาพ Visualization
├── web/                        # Prototype WeppApp
│   ├── index.html
│   ├── style.css
│   ├── app.js                     
│   └── assets/                 # เก็บไฟล์ *_triage.onnx และ config.json ที่ Export ออกมา
│
└── requirements.txt            # รายการ Library ทั้งหมดของโปรเจกต์
```

## 🚀 ลำดับขั้นตอนการรัน

เปิด Terminal แล้ว `cd` เข้ามาที่โฟลเดอร์หลักของโปรเจกต์ จากนั้นดำเนินการตามลำดับดังนี้:

### ขั้นที่ 1 · ติดตั้งสภาพแวดล้อม 
สร้าง Virtual Environment และติดตั้ง Library ทั้งหมดตามไฟล์ `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

แล้ว `cd` เข้ามาที่โฟลเดอร์ `src` ของโปรเจกต์ จากนั้นดำเนินการตามลำดับดังนี้:

### ขั้นที่ 2 · ดาวน์โหลดและเตรียมข้อมูล
รันสคริปต์เรียงตามลำดับเพื่อโหลดข้อมูล สร้างดัชนี แบ่งชุดข้อมูล ตรวจสอบความถูกต้อง และปรับขนาดภาพล่วงหน้า

```bash
# 1. ดาวน์โหลดชุดข้อมูล SIIM-ACR Pneumothorax จาก Kaggle
python dowload_data.py

# 2. สร้างไฟล์ดัชนีรวมตำแหน่งภาพ Mask และป้ายกำกับ Label
python build_registry.py

# 3. แบ่งข้อมูลเป็นชุด train / val / test
python split_data.py

# 4. ตรวจสอบสัดส่วนคลาสและแสดงภาพตัวอย่างหลังแบ่งชุดข้อมูล
python show_class_distribution.py
python visualize_splits.py

# 5. ปรับขนาดภาพ (Resize) ล่วงหน้าเพื่อลดภาระการอ่านไฟล์ระหว่างเทรน
python preprocess_data.py
```

### ขั้นที่ 3 · ตรวจสอบ Pipeline และฝึกสอนโมเดล
ทดสอบการทำงานของ DataLoader และโครงสร้างโมเดล จากนั้นเริ่มเทรน Stage 1 และ Stage 2 ตามลำดับ

```bash
# 1. ทดสอบการโหลดข้อมูลและโครงสร้าง Tensor ของโมเดล
python dataset.py
python model.py

# 2. เทรน Stage 1: Multi-Task Representation Learning (BCE + Dice Loss)
python train.py --model_name resnet34d --stage 1 --epochs 20 --batch_size 16 --lr 3e-4

# 3. เทรน Stage 2: Deep AUC Maximization (AUC Margin Loss + PESG + BSN)
python train.py --model_name resnet34d --stage 2 --epochs 15 --batch_size 16 --lr 0.01 --margin 0.8 
```

(💡 สำหรับรันต่อเนื่องข้ามคืนบน macOS: สามารถรัน Stage 1 ต่อด้วย Stage 2 และประเมินผลจบในคำสั่งเดียว พร้อมสั่ง Sleep อัตโนมัติ)

```bash
caffeinate -i sh -c "python train.py --model_name resnet34d --stage 1 --epochs 20 --batch_size 16 --lr 3e-4 && python train.py --model_name resnet34d --stage 2 --epochs 15 --batch_size 16 --lr 0.01 --margin 0.8 && python evaluate.py --model_name resnet34d --stage 2"; pmset sleepnow
```

### ขั้นที่ 4 · พล็อตกราฟ ประเมินผล และวิเคราะห์จุดผิดพลาด
สร้างกราฟประวัติการเรียนรู้ ปรับจุดตัดทางคลินิก (Sensitivity ≥ 95%) วัดผลบน Test Set และดูตัวอย่างภาพทำนายรายกลุ่ม

```bash
# 1. สร้างกราฟ Loss และ Validation Metrics ของทั้ง 2 Stage
python plot_history.py --model_name resnet34d --stage 1
python plot_history.py --model_name resnet34d --stage 2

# 2. คำนวณจุดตัดทางคลินิกและประเมินผลลัพธ์บน Test Set
python evaluate.py --model_name resnet34d --stage 2

# 3. สร้างภาพเปรียบเทียบผลทำนายแยกกลุ่ม TP / FN / FP / TN
python inference_viz.py --model_name resnet34d --stage 2
```

### ขั้นที่ 5 · ส่งออกโมเดลและเตรียมข้อมูลทดสอบหน้าเว็บ
แปลงโมเดลเป็นไฟล์ ONNX พร้อมไฟล์ตั้งค่า config.json และคัดแยกภาพจาก Test Set เพื่อนำไปทดลองอัปโหลดบน Web Prototype

```bash
# 1. แปลงไฟล์โมเดลเป็น ONNX และสร้าง config.json
python export_onnx.py --model_name resnet34d --stage 2

# 2. คัดแยกภาพตัวอย่างจาก Test Set สำหรับใช้ทดสอบใน Prototype
python prepare_demo_images.py
```

## 📊 สรุปผลการทดลอง

จากการประเมินประสิทธิภาพบนชุดข้อมูลทดสอบ (**Test Set จำนวน 1,602 ภาพ**: เคสภาวะปอดรั่ว 357 ภาพ และเคสปกติ 1,245 ภาพ) ด้วยสถาปัตยกรรม **Hybrid Segmentation + Deep AUC Maximization (Stage 2)** พบว่า **`resnet34d`** ให้ผลลัพธ์ดีที่สุดและเอาชนะโมเดลเปรียบเทียบอย่าง `densenet121` ได้ในทุกตัวชี้วัดหลัก:

### ตารางเปรียบเทียบผลลัพธ์บน Test Set

| Model | ROC-AUC | PR-AUC | Positive Dice | Mean IoU | Threshold | Sensitivity ≥ 95% | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `densenet121 hybrid(3e-4)` | `0.92` | `0.79` | **`0.33`** | `0.24` | `0.009` | `0.92` | `0.79` |
| **`resnet34d hybrid(3e-4)`** | **`0.93`** | **`0.80`** | `0.32` | `0.24` | `0.0341` | **`0.97`** | `0.70` |

### ⚠️ จุดที่ยังเป็นข้อจำกัดและแนวทางแก้ไขในอนาคต 

1. **ความละเอียดของขอบเขต Mask ยังไม่คมมาก (`Mean Pos Dice = 0.3246`)**
   * **สาเหตุ:** ภาพที่ใช้เทรนถูกย่อขนาดเหลือ `256×256` ทำให้เส้นขอบเยื่อหุ้มปอด (Visceral Pleural Line) ที่บางเพียง 1–2 พิกเซลสูญเสียรายละเอียดไปบางส่วน ประกอบกับหัว Decoder ถูกออกแบบให้เบา (`FastDecoderBlock`) เพื่อให้รันบนเว็บได้เร็ว
   * **แนวทางแก้ไข:** เพิ่มความละเอียดภาพอินพุตเป็น `512×512` และเปลี่ยน Loss ใน Stage 1 เป็น **Focal Loss + Lovász-Hinge / Tversky Loss** เพื่อบังคับให้โมเดลโฟกัสที่รอยโรคขนาดเล็ก (Small Pneumothorax) ได้แม่นยำขึ้น

2. **อัตราการแจ้งเตือนผิดพลาดในกลุ่มคนปกติยังค่อนข้างสูง (`False Positives = 369 เคส`, `PPV = 48.39%`)**
   * **สาเหตุ:** การตั้งเป้าหมาย **Sensitivity ≥ 95%** เพื่อความปลอดภัยสูงสุดทางคลินิก (กันเคสวิกฤตหลุด) ทำให้โมเดลยอมแลกกับความไวที่สูงขึ้น จนเกิดการแจ้งเตือนเกินในภาพที่มีสิ่งแปลกปลอมคล้ายเส้นขอบปอด เช่น สายท่อช่วยหายใจ (Chest Tube), ขอบกระดูกสะบัก หรือรอยพับของผิวหนัง
   * **แนวทางแก้ไข:** ใช้ **Test-Time Augmentation (TTA)** (เช่น เฉลี่ยผลทำนายจากภาพปกติและภาพกลับด้านซ้าย-ขวา) ร่วมกับการทำ **Hard Negative Mining** นำภาพกลุ่ม False Positive เหล่านี้กลับมาเทรนซ้ำในรอบถัดไป

3. **การปรับสมดุลค่าความน่าจะเป็นดิบ (Probability Calibration)**
   * **สาเหตุ:** การใช้ **Batch Score Normalization (BSN) + AUC Margin Loss** ใน Stage 2 เน้นการจัดอันดับ (Ranking) เป็นหลัก ทำให้ค่า Threshold ที่เหมาะสมถูกบีบลงมาอยู่ที่ `0.0341` แทนที่จะกระจายอยู่แถว `0.50`
   * **แนวทางแก้ไข:** เพิ่มขั้นตอน **Temperature Scaling** หรือ **Platt Scaling** บน Validation Set ก่อนส่งออกไฟล์ ONNX เพื่อให้คะแนนความน่าจะเป็นที่แสดงบนหน้าเว็บสื่อความหมายเป็นเปอร์เซ็นต์ความเสี่ยงที่แพทย์เข้าใจง่ายขึ้น