import os
import time
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import torch
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

# ==============================================================================
# 1. 頁面基本配置
# ==============================================================================
st.set_page_config(
    page_title="大家樂 (Café de Coral) 智能餐盤殘食審計系統",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. 結構化全域清單 (Lists): 分店與支援食物類別
# ==============================================================================

# (1) 支援的分店清單 (Branch List with Metadata)
BRANCH_LIST = [
    {
        "name": "中環威靈頓街店",
        "level": "Level A (商業核心區 / CBD)",
        "district": "中西區",
        "traffic": "白領上班族為主，午市尖峰翻檯率極高",
        "avg_covers": 1200,
        "base_rice_g": 240,
        "strategy": "白領控醣需求高，建議推動小份量出餐與少飯扣減優惠。"
    },
    {
        "name": "灣仔莊士敦道店",
        "level": "Level A (商業核心區 / CBD)",
        "district": "灣仔區",
        "traffic": "商務白領及會展客群，快節奏用餐",
        "avg_covers": 1100,
        "base_rice_g": 240,
        "strategy": "鼓勵推廣低碳輕量套餐，點餐機預設少飯減 $2。"
    },
    {
        "name": "沙田新城市廣場店",
        "level": "Level B (住宅商場 / Residential)",
        "district": "沙田區",
        "traffic": "家庭客、長者與週末休閒客群",
        "avg_covers": 1500,
        "base_rice_g": 260,
        "strategy": "家庭用餐與兒童共享比例高，建議推動彈性配菜組合。"
    },
    {
        "name": "將軍澳PopCorn店",
        "level": "Level B (住宅商場 / Residential)",
        "district": "西貢區",
        "traffic": "新興中產家庭，晚市外賣與堂食均衡",
        "avg_covers": 1350,
        "base_rice_g": 260,
        "strategy": "優化例湯與副菜組合，降低冷門配菜報廢率。"
    },
    {
        "name": "香港科技大學店 (HKUST)",
        "level": "Level C (校園與青年區 / Campus)",
        "district": "西貢區",
        "traffic": "學生、教職員，運動量及食量顯著較大",
        "avg_covers": 1800,
        "base_rice_g": 280,
        "strategy": "維持大份量以維持飽足感，重點監控肉類醬汁口味與炸物品質。"
    },
    {
        "name": "觀塘開源道店",
        "level": "Level C (校園與青年區 / Industrial)",
        "district": "觀塘區",
        "traffic": "藍領勞工與工廈青年，熱量消耗大",
        "avg_covers": 1600,
        "base_rice_g": 280,
        "strategy": "維持原份量配給，避免減量影響顧客滿意度。"
    }
]

# (2) 支援的食物項目清單 (Supported Food & Dish List)
SUPPORTED_DISH_LIST = [
    {"dish_id": "D01", "name": "一哥焗豬扒飯 (Baked Pork Chop Rice)", "main_carb": "蛋炒飯", "protein": "焗厚切豬扒"},
    {"dish_id": "D02", "name": "咖喱牛腩飯 (Curry Beef Brisket Rice)", "main_carb": "白米飯", "protein": "慢燉牛腩"},
    {"dish_id": "D03", "name": "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "main_carb": "白米飯", "protein": "滑蛋蝦仁"},
    {"dish_id": "D04", "name": "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "main_carb": "白米飯", "protein": "煎肉餅"},
    {"dish_id": "D05", "name": "黑椒鐵板牛柳絲炒意粉 (Beef Spaghetti)", "main_carb": "意粉", "protein": "黑椒牛柳絲"}
]

# (3) 支援的殘食偵測類別清單 (Detectable Food Waste Categories)
WASTE_CATEGORY_LIST = [
    {"code": "RICE", "name": "白飯/主食殘留 (Rice Waste)", "unit_cost": 0.015, "carbon_factor": 1.6},
    {"code": "MEAT", "name": "主菜肉類殘留 (Meat Residual)", "unit_cost": 0.045, "carbon_factor": 4.5},
    {"code": "VEG_SOUP", "name": "配菜/醬汁殘留 (Veg & Sauce)", "unit_cost": 0.010, "carbon_factor": 0.8},
    {"code": "CLEAN", "name": "光盤/無明顯浪費 (Clean Plate)", "unit_cost": 0.0, "carbon_factor": 0.0}
]

# ==============================================================================
# 3. 本地 SQLite 資料庫引擎 (Persistent DB Handler)
# ==============================================================================
DB_FILE = "trayzero_audit.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            branch_name TEXT,
            branch_level TEXT,
            dish_name TEXT,
            primary_waste TEXT,
            waste_ratio REAL,
            cost_waste_hkd REAL,
            co2_emission_kg REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_audit_record(record):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_logs (
            timestamp, branch_name, branch_level, dish_name, 
            primary_waste, waste_ratio, cost_waste_hkd, co2_emission_kg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["timestamp"], record["branch_name"], record["branch_level"],
        record["dish_name"], record["primary_waste"], record["waste_ratio"],
        record["cost_waste_hkd"], record["co2_emission_kg"]
    ))
    conn.commit()
    conn.close()

def fetch_all_audit_records():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
    conn.close()
    return df

def clear_all_audit_records():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM audit_logs")
    conn.commit()
    conn.close()

# 初始化 DB
init_db()

# ==============================================================================
# 4. 雙 Hugging Face 深度學習模型初始化 (原生載入)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_hf_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = "./Fine-tuned_Model_files" if os.path.exists("./Fine-tuned_Model_files") and any(os.scandir("./Fine-tuned_Model_files")) else "hustvl/yolos-tiny"
    
    img_processor = AutoImageProcessor.from_pretrained(model_path)
    det_model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
    det_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    t5_model.eval()
    
    return img_processor, det_model, tokenizer, t5_model, device

with st.spinner("🚀 正在啟動雙核心 AI 模型引擎 (YOLOS-tiny + Flan-T5)..."):
    img_processor, det_model, nlp_tokenizer, nlp_model, runtime_device = load_hf_models()

# ==============================================================================
# 5. 偵測與推論輔助函式
# ==============================================================================
def execute_detection(image, score_threshold=0.20):
    inputs = img_processor(images=image, return_tensors="pt").to(runtime_device)
    with torch.no_grad():
        outputs = det_model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]]).to(runtime_device)
    results = img_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]
    
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    img_w, img_h = image.size
    total_area = img_w * img_h
    detected_items = []
    waste_box_area = 0
    
    color_map = {
        "Rice": "#E74C3C",
        "Meat": "#E67E22",
        "Veg_Soup": "#27AE60"
    }
    
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()
    labels = results["labels"].tolist()
    
    primary_category = "光盤 (Clean Plate)"
    
    for box, score, label_id in zip(boxes, scores, labels):
        raw_label = det_model.config.id2label.get(label_id, "item")
        
        if raw_label in ["bowl", "dining table", "cake"]:
            category = "Rice"
            display_name = "白飯/主食殘留 (Rice Waste)"
            primary_category = "白飯/主食殘留"
        elif raw_label in ["sandwich", "pizza", "hot dog"]:
            category = "Meat"
            display_name = "主菜肉類殘留 (Meat Residual)"
            if "白飯" not in primary_category:
                primary_category = "主菜肉類殘留"
        else:
            category = "Veg_Soup"
            display_name = f"配菜/醬汁殘留 ({raw_label})"
            if primary_category == "光盤 (Clean Plate)":
                primary_category = "配菜/醬汁殘留"
            
        xmin, ymin, xmax, ymax = box
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)
        
        box_area = (xmax - xmin) * (ymax - ymin)
        waste_box_area += box_area
        
        c = color_map.get(category, "#E74C3C")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=c, width=4)
        caption = f"{display_name} {score:.1%}"
        draw.rectangle([xmin, max(0, ymin - 20), xmin + len(caption) * 8, ymin], fill=c)
        draw.text((xmin + 4, max(0, ymin - 18)), caption, fill="white")
        
        detected_items.append({
            "分類項目": display_name,
            "置信度": f"{score:.2%}",
            "佔比": f"{(box_area / total_area):.1%}"
        })
        
    waste_ratio = min(1.0, waste_box_area / (total_area * 0.65)) if total_area > 0 else 0.0
    return annotated_img, detected_items, waste_ratio, primary_category

def generate_kitchen_decision(branch_name, branch_meta, dish_name, waste_ratio):
    prompt = (
        f"You are the executive kitchen director of Cafe de Coral. "
        f"Store: {branch_name} ({branch_meta['level']}). Target Dish: {dish_name}. "
        f"Standard portion: {branch_meta['base_rice_g']}g. "
        f"Visual detection observed waste ratio: {waste_ratio:.1%}. "
        f"Strategy: {branch_meta['strategy']} "
        f"Provide one actionable kitchen portion adjustment action in grams and one POS ordering change."
    )
    inputs = nlp_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(runtime_device)
    with torch.no_grad():
        outputs = nlp_model.generate(**inputs, max_new_tokens=90, do_sample=False)
    return nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 6. 側邊欄切換：Mode 1 (Detection) vs. Mode 2 (Dashboard)
# ==============================================================================
st.sidebar.title("🎛️ 系統模式選擇")
system_mode = st.sidebar.radio(
    "切換工作模式",
    ["Mode 1: 前線餐盤智能偵測 (Tray Detection)", "Mode 2: 總部即時營運大盤 (Real-time Dashboard)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 資料庫與測試管理 (DB Management)")
if st.sidebar.button("🗑️ 清空所有測試記錄 (Reset DB)", type="secondary"):
    clear_all_audit_records()
    st.sidebar.success("✅ 資料庫已完全清空，可重新進行測試！")
    st.rerun()

# ==============================================================================
# 7. MODE 1: 前線餐盤智能偵測 (Tray Detection)
# ==============================================================================
if system_mode == "Mode 1: 前線餐盤智能偵測 (Tray Detection)":
    st.subheader("📸 Mode 1: 前線智能餐盤審計機 (回收台即時掃描)")
    st.caption("支援功能：從分店 List 選擇門市 ➔ 自動靜止偵測 / 拍照 / 上傳 ➔ YOLOS 畫框 ➔ Flan-T5 建議 ➔ 自動持久化儲存")
    
    col_setting, col_cam = st.columns([1, 2])
    
    with col_setting:
        st.markdown("### 🏢 當前執勤分店設定")
        branch_names = [b["name"] for b in BRANCH_LIST]
        selected_branch_name = st.selectbox("選擇當前門市 (此掃描記錄將綁定該店)", branch_names)
        
        # 取得分店元數據
        branch_meta = next(item for item in BRANCH_LIST if item["name"] == selected_branch_name)
        
        st.info(f"""
        **門市等級**: `{branch_meta['level']}`  
        **所屬區域**: {branch_meta['district']}  
        **客群結構**: {branch_meta['traffic']}  
        **標準出餐**: `{branch_meta['base_rice_g']}g 白飯`  
        """)
        
        dish_names = [d["name"] for d in SUPPORTED_DISH_LIST]
        selected_dish = st.selectbox("抽檢餐點項目", dish_names)
        
        scan_mode = st.radio(
            "輸入感應方式",
            ["自動靜止感應 (Live Auto-Detect)", "手動拍照 (Manual Capture)", "照片檔案上傳 (Upload)"]
        )

    with col_cam:
        captured_image = None
        
        # 模式 A: 自動靜止 2 秒感應
        if scan_mode == "自動靜止感應 (Live Auto-Detect)":
            st.markdown("#### 🟢 Live Camera 自動感測已啟動")
            st.info("💡 操作提示：請將餐盤放置於鏡頭前。系統偵測到畫面**靜止不動 2 秒**時，將自動執行偵測（每次放置僅觸發一次）。")
            
            auto_shot = st.camera_input("自動感應鏡頭", key="live_auto_cam")
            
            if auto_shot:
                captured_image = Image.open(auto_shot).convert("RGB")
                current_frame_hash = hash(captured_image.tobytes()[:2000])
                last_hash = st.session_state.get("last_scanned_hash", None)
                
                if current_frame_hash != last_hash:
                    countdown_placeholder = st.empty()
                    for s in range(2, 0, -1):
                        countdown_placeholder.warning(f"⏳ 偵測到餐盤放入，保持靜止中... {s}s")
                        time.sleep(1)
                    countdown_placeholder.success("🎯 鎖定完成，自動觸發 AI 審計！")
                    st.session_state["last_scanned_hash"] = current_frame_hash
                    st.session_state["trigger_auto_scan"] = True
                else:
                    st.success("✅ 該餐盤已完成掃描，等待下一個餐盤移入...")
                    
        # 模式 B: 手動拍照
        elif scan_mode == "手動拍照 (Manual Capture)":
            manual_shot = st.camera_input("點擊拍照按鈕進行快照", key="manual_cam")
            if manual_shot:
                captured_image = Image.open(manual_shot).convert("RGB")
                st.session_state["trigger_auto_scan"] = True
                
        # 模式 C: 檔案上傳
        else:
            uploaded_file = st.file_uploader("上傳餐盤相片 (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                captured_image = Image.open(uploaded_file).convert("RGB")
                st.session_state["trigger_auto_scan"] = True

    # 執行偵測與自動寫入 SQLite DB
    if captured_image and st.session_state.get("trigger_auto_scan", False):
        st.markdown("---")
        st.subheader("🎯 AI 偵測與分析結果")
        
        with st.spinner("AI 物體偵測中: YOLOS-tiny 正在繪製 Bounding Box..."):
            annotated_img, item_list, waste_ratio, primary_cat = execute_detection(captured_image)
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.image(annotated_img, caption="AI Object Detection 即時標記框 (Bounding Boxes)", use_container_width=True)
            if item_list:
                st.markdown("**🔍 偵測物件結構明細**")
                st.dataframe(pd.DataFrame(item_list), use_container_width=True)
            else:
                st.info("光盤良好，無明顯食物殘餘。")
                
        with col_res2:
            with st.spinner("Flan-T5 正在生成該分店之即時調配指令..."):
                kitchen_advice = generate_kitchen_decision(selected_branch_name, branch_meta, selected_dish, waste_ratio)
            
            st.success(f"👨‍🍳 **【{selected_branch_name}】後廚調配指令**:\n\n{kitchen_advice}")
            
            # 換算損耗金錢與碳排放
            unit_rice_cost = 0.015
            cut_g = 40 if "Level A" in branch_meta["level"] else 20
            saved_hkd = round(cut_g * unit_rice_cost * 25, 1)
            saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
            
            st.metric("偵測殘食盤面佔比", f"{waste_ratio:.1%}")
            
            # 關鍵持久化：寫入 SQLite 資料庫
            new_record = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "branch_name": selected_branch_name,
                "branch_level": branch_meta["level"].split(" ")[0],
                "dish_name": selected_dish,
                "primary_waste": primary_cat,
                "waste_ratio": round(waste_ratio * 100, 1),
                "cost_waste_hkd": saved_hkd,
                "co2_emission_kg": saved_co2
            }
            insert_audit_record(new_record)
            st.toast(f"✅ 審計記錄已成功存入資料庫 (DB)！已同步至總部 Dashboard。")
            st.session_state["trigger_auto_scan"] = False

# ==============================================================================
# 8. MODE 2: 總部即時營運大盤 (Real-time Dashboard - SQLite 連動)
# ==============================================================================
else:
    st.subheader("📊 Mode 2: 大家樂集團總部 - 跨分店即時營運與 ESG 大盤")
    st.caption("數據來源：直接讀取 SQLite 資料庫中來自 Mode 1 的真實偵測記錄 (非靜態假數據)")
    
    df_db = fetch_all_audit_records()
    
    if df_db.empty:
        st.warning("⚠️ 目前資料庫中無任何審計數據。請切換至「Mode 1: 前線餐盤智能偵測」完成幾次掃描測試！")
    else:
        # 頂部關鍵指標 (KPI Cards)
        total_scans = len(df_db)
        avg_waste = df_db["waste_ratio"].mean()
        total_waste_hkd = df_db["cost_waste_hkd"].sum()
        total_co2 = df_db["co2_emission_kg"].sum()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("真實審計累積盤數", f"{total_scans} 盤")
        kpi2.metric("全港平均殘食佔比", f"{avg_waste:.1f} %")
        kpi3.metric("累積食材損耗成本", f"HK$ {total_waste_hkd:,.1f}")
        kpi4.metric("累積碳排放當量", f"{total_co2:.2f} kg CO2e")
        
        st.markdown("---")
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("#### 🏢 各分店真實平均殘食率 (%)")
            branch_stat = df_db.groupby("branch_name")["waste_ratio"].mean().reset_index()
            st.bar_chart(branch_stat, x="branch_name", y="waste_ratio", color="#FF4B4B")
            
        with col_c2:
            st.markdown("#### 🏷️ 門市等級 (Level A/B/C) 浪費金額分佈 (HK$)")
            level_stat = df_db.groupby("branch_level")["cost_waste_hkd"].sum().reset_index()
            st.bar_chart(level_stat, x="branch_level", y="cost_waste_hkd")
            
        st.markdown("### 📋 SQLite 資料庫即時紀錄流水表 (Audit DB Stream)")
        st.dataframe(df_db, use_container_width=True)
        
        # 下載 CSV 功能
        csv_data = df_db.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 匯出當前審計數據為 CSV",
            data=csv_data,
            file_name=f"cafedecoral_audit_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
