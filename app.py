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
# 1. 頁面基本配置與 Session State 初始化
# ==============================================================================
st.set_page_config(
    page_title="大家樂 (Café de Coral) 智能餐盤殘食審計系統",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化最新偵測快取 (跨刷新常駐顯示)
if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None

if "last_scanned_hash" not in st.session_state:
    st.session_state["last_scanned_hash"] = None

# ==============================================================================
# 2. 基礎資料檔案與 SQLite 資料庫
# ==============================================================================
BRANCH_FILE = "master_branches.csv"
DISH_FILE = "master_dishes.csv"
DB_FILE = "trayzero_audit.db"

DEFAULT_BRANCHES = [
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
        "name": "沙田新城市廣場店",
        "level": "Level B (住宅商場 / Residential)",
        "district": "沙田區",
        "traffic": "家庭客、長者與週末休閒客群",
        "avg_covers": 1500,
        "base_rice_g": 260,
        "strategy": "家庭用餐與兒童共享比例高，建議推動彈性配菜組合。"
    },
    {
        "name": "香港科技大學店 (HKUST)",
        "level": "Level C (校園與青年區 / Campus)",
        "district": "西貢區",
        "traffic": "學生、教職員，運動量及食量顯著較大",
        "avg_covers": 1800,
        "base_rice_g": 280,
        "strategy": "維持大份量以維持飽足感，重點監控肉類醬汁口味與炸物品質。"
    }
]

DEFAULT_DISHES = [
    {"dish_id": "D01", "name": "一哥焗豬扒飯 (Baked Pork Chop Rice)", "main_carb": "蛋炒飯", "protein": "焗厚切豬扒"},
    {"dish_id": "D02", "name": "咖喱牛腩飯 (Curry Beef Brisket Rice)", "main_carb": "白米飯", "protein": "慢燉牛腩"},
    {"dish_id": "D03", "name": "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "main_carb": "白米飯", "protein": "滑蛋蝦仁"},
    {"dish_id": "D04", "name": "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "main_carb": "白米飯", "protein": "煎肉餅"}
]

def load_master_data():
    if os.path.exists(BRANCH_FILE):
        df_b = pd.read_csv(BRANCH_FILE)
    else:
        df_b = pd.DataFrame(DEFAULT_BRANCHES)
        df_b.to_csv(BRANCH_FILE, index=False)

    if os.path.exists(DISH_FILE):
        df_d = pd.read_csv(DISH_FILE)
    else:
        df_d = pd.DataFrame(DEFAULT_DISHES)
        df_d.to_csv(DISH_FILE, index=False)
    return df_b, df_d

df_branches, df_dishes = load_master_data()

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

init_db()

# ==============================================================================
# 3. 雙 Hugging Face 深度學習模型初始化 (原生類別載入)
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

with st.spinner("🚀 正在啟動雙核心 AI 審計引擎 (YOLOS-tiny + Flan-T5)..."):
    img_processor, det_model, nlp_tokenizer, nlp_model, runtime_device = load_hf_models()

# ==============================================================================
# 4. 偵測與推論輔助函式
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

def generate_kitchen_decision(branch_name, branch_level, base_rice_g, strategy, dish_name, waste_ratio):
    prompt = (
        f"You are the executive kitchen director of Cafe de Coral. "
        f"Store: {branch_name} ({branch_level}). Target Dish: {dish_name}. "
        f"Standard portion: {base_rice_g}g. "
        f"Visual detection observed waste ratio: {waste_ratio:.1%}. "
        f"Strategy: {strategy} "
        f"Provide one actionable kitchen portion adjustment action in grams and one POS ordering change."
    )
    inputs = nlp_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(runtime_device)
    with torch.no_grad():
        outputs = nlp_model.generate(**inputs, max_new_tokens=90, do_sample=False)
    return nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 5. 側邊欄切換：Mode 1, Mode 2, Mode 3
# ==============================================================================
st.sidebar.title("🎛️ 系統模式選擇")
system_mode = st.sidebar.radio(
    "切換工作模式",
    [
        "Mode 1: 前線餐盤智能偵測 (Tray Detection)", 
        "Mode 2: 總部即時營運大盤 (Real-time Dashboard)",
        "Mode 3: 基礎資料設定與上傳 (Master Data Management)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 測試管理工具")
if st.sidebar.button("🗑️ 清空所有審計記錄 (Reset DB)", type="secondary"):
    clear_all_audit_records()
    st.session_state["latest_result"] = None
    st.session_state["last_scanned_hash"] = None
    st.sidebar.success("✅ 資料庫與最新快取已清空！")
    st.rerun()

# ==============================================================================
# 6. MODE 1: 前線餐盤智能偵測 (Tray Detection with Persistent Live Cam)
# ==============================================================================
if system_mode == "Mode 1: 前線餐盤智能偵測 (Tray Detection)":
    st.subheader("📸 Mode 1: 前線餐盤智慧審計機 (Live Camera 常駐掃描)")
    st.caption("Live Camera 保持常駐監控 ➔ 餐盤靜止 2 秒自動鎖定 ➔ 雙 Pipeline 運算 ➔ 右側即時常駐顯示 Latest Result")

    # 左右排版：左邊為相機與操作區，右邊為最新偵測結果看板 (Latest Result)
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        # 分店與餐點選擇
        st.markdown("##### 🏢 審計門市與餐點配置")
        c_b, c_d = st.columns(2)
        with c_b:
            selected_branch_name = st.selectbox("執勤門市", df_branches["name"].tolist())
            b_row = df_branches[df_branches["name"] == selected_branch_name].iloc[0]
        with c_d:
            selected_dish = st.selectbox("抽檢餐點", df_dishes["name"].tolist())

        st.caption(f"門市等級: `{b_row['level']}` | 標準飯量: `{b_row['base_rice_g']}g` | 區域: `{b_row['district']}`")

        # 輸入方式選擇
        scan_mode = st.radio(
            "相機與輸入模式",
            ["🟢 Live Camera 長開 (靜止自動偵測)", "📸 手動快照模式 (Manual Snapshot)", "📁 照片檔案上傳 (Upload)"],
            horizontal=True
        )

        captured_image = None
        should_run_detection = False

        if scan_mode == "🟢 Live Camera 長開 (靜止自動偵測)":
            st.markdown("**即時視頻流 (Live Camera 保持開啟中)**")
            live_shot = st.camera_input("持續監控畫面", key="permanent_live_cam")
            
            if live_shot:
                captured_image = Image.open(live_shot).convert("RGB")
                current_frame_hash = hash(captured_image.tobytes()[:3000])
                
                # 比對雜湊判定畫面是否靜止且未重複掃描
                if current_frame_hash != st.session_state["last_scanned_hash"]:
                    countdown_box = st.empty()
                    for s in range(2, 0, -1):
                        countdown_box.warning(f"⏳ 偵測到餐盤放入，畫面靜止鎖定中... {s} 秒")
                        time.sleep(1)
                    countdown_box.success("🎯 鎖定完成！自動觸發 AI 審計分析...")
                    st.session_state["last_scanned_hash"] = current_frame_hash
                    should_run_detection = True
                else:
                    st.info("🟢 監控中：當前餐盤已完成分析。更換餐盤將自動觸發下一次偵測。")

        elif scan_mode == "📸 手動快照模式 (Manual Snapshot)":
            manual_shot = st.camera_input("手動拍照", key="manual_cam_shot")
            if manual_shot:
                captured_image = Image.open(manual_shot).convert("RGB")
                should_run_detection = True

        else:
            up_file = st.file_uploader("上傳餐盤相片 (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if up_file:
                captured_image = Image.open(up_file).convert("RGB")
                if st.button("🚀 執行上傳相片偵測", type="primary"):
                    should_run_detection = True

        # 執行偵測核心並更新 Latest Result
        if captured_image and should_run_detection:
            with st.spinner("Pipeline 1 & 2 運算中: YOLOS-tiny 正在繪製 Bounding Box..."):
                annotated_img, item_list, waste_ratio, primary_cat = execute_detection(captured_image)
                kitchen_advice = generate_kitchen_decision(
                    selected_branch_name, b_row["level"], b_row["base_rice_g"], 
                    b_row["strategy"], selected_dish, waste_ratio
                )

            # 財務與碳排計算
            unit_rice_cost = 0.015
            cut_g = 40 if "Level A" in str(b_row["level"]) else 20
            saved_hkd = round(cut_g * unit_rice_cost * 25, 1)
            saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 寫入 SQLite 資料庫
            db_record = {
                "timestamp": timestamp_str,
                "branch_name": selected_branch_name,
                "branch_level": str(b_row["level"]).split(" ")[0],
                "dish_name": selected_dish,
                "primary_waste": primary_cat,
                "waste_ratio": round(waste_ratio * 100, 1),
                "cost_waste_hkd": saved_hkd,
                "co2_emission_kg": saved_co2
            }
            insert_audit_record(db_record)

            # 更新 Session State 中的 Latest Result
            st.session_state["latest_result"] = {
                "image": annotated_img,
                "item_list": item_list,
                "waste_ratio": waste_ratio,
                "primary_cat": primary_cat,
                "kitchen_advice": kitchen_advice,
                "branch_name": selected_branch_name,
                "dish_name": selected_dish,
                "timestamp": timestamp_str,
                "cost": saved_hkd,
                "co2": saved_co2
            }
            st.toast("✅ 偵測完成！最新結果已更新並同步至資料庫。")

    # 右側：最新偵測結果看板 (Latest Detection Result)
    with col_right:
        st.markdown("### 🎯 最新偵測結果 (Latest Detection Result)")
        latest = st.session_state["latest_result"]

        if latest is None:
            st.info("💡 尚未執行偵測。請保持 Live Camera 開啟並放置餐盤，或手動拍攝/上傳。")
            # 預設佔位圖
            placeholder_img = Image.new("RGB", (400, 300), color=(240, 240, 240))
            d = ImageDraw.Draw(placeholder_img)
            d.text((120, 140), "等待餐盤輸入中...", fill=(150, 150, 150))
            st.image(placeholder_img, caption="即時預覽看板", use_container_width=True)
        else:
            # 顯示最新標籤圖
            st.image(latest["image"], caption=f"最新審計影像 ({latest['timestamp']})", use_container_width=True)

            # 核心指標卡片
            m1, m2, m3 = st.columns(3)
            m1.metric("殘食佔比", f"{latest['waste_ratio']:.1%}")
            m2.metric("主要殘留", latest["primary_cat"])
            m3.metric("損耗金額", f"HK$ {latest['cost']}")

            # 後廚即時行動卡片
            st.success(f"👨‍🍳 **後廚出餐校準指令 ({latest['branch_name']})**:\n\n{latest['kitchen_advice']}")

            # 物件明細
            if latest["item_list"]:
                with st.expander("查看 Bounding Box 偵測物件明細", expanded=False):
                    st.dataframe(pd.DataFrame(latest["item_list"]), use_container_width=True)

# ==============================================================================
# 7. MODE 2: 總部即時營運大盤 (Real-time Dashboard - SQLite 連動)
# ==============================================================================
elif system_mode == "Mode 2: 總部即時營運大盤 (Real-time Dashboard)":
    st.subheader("📊 Mode 2: 大家樂集團總部 - 跨分店即時營運與 ESG 大盤")
    st.caption("數據來源：直接讀取 SQLite 資料庫中來自 Mode 1 的真實偵測記錄")
    
    df_db = fetch_all_audit_records()
    
    if df_db.empty:
        st.warning("⚠️ 目前資料庫中無任何審計數據。請切換至「Mode 1」完成幾次掃描測試！")
    else:
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
            
        st.markdown("### 📋 SQLite 資料庫即時紀錄流水表")
        st.dataframe(df_db, use_container_width=True)
        
        csv_data = df_db.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 匯出當前審計數據為 CSV",
            data=csv_data,
            file_name=f"cafedecoral_audit_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )

# ==============================================================================
# 8. MODE 3: 基礎資料設定與上傳 (Master Data Management)
# ==============================================================================
else:
    st.subheader("⚙️ Mode 3: 基礎資料管理 (分店清單 & 餐點品項 CSV 上傳)")
    st.caption("在此維護分店清單與餐點清單。支援上傳 CSV 批量更新、下載範本，或直接在網頁表格中手動修改。")

    tab_branch_mgt, tab_dish_mgt = st.tabs(["🏢 分店清單管理 (Branch List)", "🍱 餐點品項管理 (Dish List)"])

    with tab_branch_mgt:
        st.markdown("#### 1. 上傳分店 CSV 檔案 (Batch Upload)")
        col_b_up, col_b_dl = st.columns([2, 1])
        with col_b_dl:
            sample_branch_csv = pd.DataFrame(DEFAULT_BRANCHES).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下載分店 CSV 格式範本",
                data=sample_branch_csv,
                file_name="branches_template.csv",
                mime="text/csv"
            )
        with col_b_up:
            branch_upload = st.file_uploader("選擇分店 CSV 進行上傳覆蓋", type=["csv"], key="branch_uploader")
            if branch_upload is not None:
                try:
                    uploaded_df = pd.read_csv(branch_upload)
                    required_cols = {"name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"}
                    if required_cols.issubset(uploaded_df.columns):
                        uploaded_df.to_csv(BRANCH_FILE, index=False)
                        st.success(f"🎉 成功更新 {len(uploaded_df)} 間分店資料！")
                        st.rerun()
                    else:
                        st.error(f"❌ 格式錯誤，必須包含欄位：{required_cols}")
                except Exception as e:
                    st.error(f"❌ 讀取失敗: {e}")

        st.markdown("#### 2. 線上手動檢視與直接編輯 (Live Data Editor)")
        edited_branches = st.data_editor(df_branches, num_rows="dynamic", use_container_width=True, key="branch_editor")
        if st.button("💾 儲存分店手動修改內容", type="primary"):
            edited_branches.to_csv(BRANCH_FILE, index=False)
            st.success("✅ 分店清單已更新！")
            st.rerun()

    with tab_dish_mgt:
        st.markdown("#### 1. 上傳餐點品項 CSV 檔案 (Batch Upload)")
        col_d_up, col_d_dl = st.columns([2, 1])
        with col_d_dl:
            sample_dish_csv = pd.DataFrame(DEFAULT_DISHES).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 下載餐點 CSV 格式範本",
                data=sample_dish_csv,
                file_name="dishes_template.csv",
                mime="text/csv"
            )
        with col_d_up:
            dish_upload = st.file_uploader("選擇餐點 CSV 進行上傳覆蓋", type=["csv"], key="dish_uploader")
            if dish_upload is not None:
                try:
                    uploaded_dish_df = pd.read_csv(dish_upload)
                    required_dish_cols = {"dish_id", "name", "main_carb", "protein"}
                    if required_dish_cols.issubset(uploaded_dish_df.columns):
                        uploaded_dish_df.to_csv(DISH_FILE, index=False)
                        st.success(f"🎉 成功更新 {len(uploaded_dish_df)} 項餐點資料！")
                        st.rerun()
                    else:
                        st.error(f"❌ 格式錯誤，必須包含欄位：{required_dish_cols}")
                except Exception as e:
                    st.error(f"❌ 讀取失敗: {e}")

        st.markdown("#### 2. 線上手動檢視與直接編輯 (Live Data Editor)")
        edited_dishes = st.data_editor(df_dishes, num_rows="dynamic", use_container_width=True, key="dish_editor")
        if st.button("💾 儲存餐點手動修改內容", type="primary"):
            edited_dishes.to_csv(DISH_FILE, index=False)
            st.success("✅ 餐點清單已更新！")
            st.rerun()
