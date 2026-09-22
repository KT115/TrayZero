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
    AutoModelForSeq2SeqLM,
    pipeline
)

# ==============================================================================
# 0. 全域常數設定 (Global Constants & Defaults)
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


# ==============================================================================
# 1. 資料庫與持久化函數群 (Database & Master Data Management)
# ==============================================================================
def init_sqlite_db(db_path: str = DB_FILE) -> None:
    """初始化審計數據庫架構"""
    with sqlite3.connect(db_path) as conn:
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


def save_audit_record(record: dict, db_path: str = DB_FILE) -> None:
    """插入單筆審計結果到資料庫"""
    with sqlite3.connect(db_path) as conn:
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


def get_all_audit_records(db_path: str = DB_FILE) -> pd.DataFrame:
    """查詢所有歷史審計資料"""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)


def truncate_audit_db(db_path: str = DB_FILE) -> None:
    """清空資料庫所有記錄"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM audit_logs")
        conn.commit()


def load_master_meta() -> tuple[pd.DataFrame, pd.DataFrame]:
    """載入或初始化分店與菜品基礎設定"""
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


# ==============================================================================
# 2. 深度學習模型引擎 (Hugging Face Pipelines & Native Models)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def init_ai_pipeline_engine():
    """載入視覺偵測、文字生成與零樣本分類模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./Fine-tuned_Model_files" if os.path.exists("./Fine-tuned_Model_files") and any(os.scandir("./Fine-tuned_Model_files")) else "hustvl/yolos-tiny"
    
    # 1. 物件偵測模型 (YOLOS)
    img_processor = AutoImageProcessor.from_pretrained(model_path)
    det_model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
    det_model.eval()
    
    # 2. 決策生成模型 (Flan-T5)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    t5_model.eval()
    
    # 3. 零樣本菜品識別 (CLIP)
    clip_classifier = pipeline(
        "zero-shot-image-classification", 
        model="openai/clip-vit-base-patch32", 
        device=0 if torch.cuda.is_available() else -1
    )
    
    return {
        "processor": img_processor,
        "detector": det_model,
        "tokenizer": tokenizer,
        "generator": t5_model,
        "clip": clip_classifier,
        "device": device
    }


def predict_dish_category(image: Image.Image, candidate_labels: list[str], engine: dict) -> tuple[str, float]:
    """透過 CLIP Zero-Shot 分類自動識別托盤上的主力餐點"""
    results = engine["clip"](image, candidate_labels=candidate_labels)
    return results[0]["label"], results[0]["score"]


def run_tray_waste_detection(image: Image.Image, engine: dict, threshold: float = 0.20) -> tuple[Image.Image, list[dict], float, str]:
    """執行 YOLOS 目標偵測並繪製邊界框"""
    inputs = engine["processor"](images=image, return_tensors="pt").to(engine["device"])
    with torch.no_grad():
        outputs = engine["detector"](**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]]).to(engine["device"])
    results = engine["processor"].post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    img_w, img_h = image.size
    total_area = img_w * img_h
    detected_items = []
    waste_box_area = 0
    
    color_map = {"Rice": "#E74C3C", "Meat": "#E67E22", "Veg_Soup": "#27AE60"}
    primary_category = "光盤 (Clean Plate)"
    
    for box, score, label_id in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()):
        raw_label = engine["detector"].config.id2label.get(label_id, "item")
        
        if raw_label in ["bowl", "dining table", "cake"]:
            category, display_name = "Rice", "白飯/主食殘留 (Rice Waste)"
            primary_category = "白飯/主食殘留"
        elif raw_label in ["sandwich", "pizza", "hot dog"]:
            category, display_name = "Meat", "主菜肉類殘留 (Meat Residual)"
            if "白飯" not in primary_category:
                primary_category = "主菜肉類殘留"
        else:
            category, display_name = "Veg_Soup", f"配菜/醬汁殘留 ({raw_label})"
            if primary_category == "光盤 (Clean Plate)":
                primary_category = "配菜/醬汁殘留"
            
        xmin, ymin = max(0, box[0]), max(0, box[1])
        xmax, ymax = min(img_w, box[2]), min(img_h, box[3])
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


def generate_portioning_directive(
    branch_name: str, branch_level: str, base_rice_g: float, 
    strategy: str, dish_name: str, waste_ratio: float, engine: dict
) -> str:
    """透過 Flan-T5 依據分店等級與殘留率動態生成後廚行動建議"""
    prompt = (
        f"You are the executive kitchen director of Cafe de Coral. "
        f"Store: {branch_name} ({branch_level}). Target Dish: {dish_name}. "
        f"Standard portion: {base_rice_g}g. "
        f"Visual detection observed waste ratio: {waste_ratio:.1%}. "
        f"Strategy: {strategy} "
        f"Provide one actionable kitchen portion adjustment action in grams and one POS ordering change."
    )
    inputs = engine["tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(engine["device"])
    with torch.no_grad():
        outputs = engine["generator"].generate(**inputs, max_new_tokens=90, do_sample=False)
    return engine["tokenizer"].decode(outputs[0], skip_special_tokens=True)


def calculate_cost_and_esg(waste_ratio: float, branch_level: str) -> tuple[float, float]:
    """計算食材損失成本與碳排放當量"""
    unit_rice_cost = 0.015
    cut_g = 40 if "Level A" in str(branch_level) else 20
    saved_hkd = round(cut_g * unit_rice_cost * 25, 1)
    saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
    return saved_hkd, saved_co2


# ==============================================================================
# 3. 畫面渲染函數群 (UI View Components)
# ==============================================================================
def render_header():
    """渲染主標題與架構說明"""
    st.title("🍽️ 大家樂 (Café de Coral) 智能餐盤殘食審計與中央調配系統")
    st.markdown(
        "**ISOM5240 Group Project** | 雙 Pipeline 深度學習架構: "
        "`YOLOS-tiny (Object Detection)` $\\rightarrow$ `Flan-T5 (Context Decision Engine)`"
    )


def render_sidebar_controls() -> str:
    """渲染側邊欄模式選擇與除錯工具"""
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
        truncate_audit_db()
        st.session_state["latest_result"] = None
        st.session_state["last_scanned_hash"] = None
        st.sidebar.success("✅ 資料庫與快取已清空！")
        st.rerun()
    return system_mode


def render_mode_1_detection(df_branches: pd.DataFrame, df_dishes: pd.DataFrame, engine: dict):
    """Mode 1: 前線餐盤智能偵測與 Live 監控"""
    st.subheader("📸 Mode 1: 前線餐盤智慧審計機 (Live Camera 常駐掃描 + 餐點自動辨識)")
    st.caption("Live Camera 保持監控 ➔ 畫面靜止 2 秒自動鎖定 ➔ AI 自動識別餐點與殘食 ➔ 同步至總部大盤")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("##### 🏢 執勤分店與餐點辨識配置")
        selected_branch_name = st.selectbox("執勤門市", df_branches["name"].tolist())
        b_row = df_branches[df_branches["name"] == selected_branch_name].iloc[0]
        st.caption(f"門市等級: `{b_row['level']}` | 標準飯量: `{b_row['base_rice_g']}g` | 區域: `{b_row['district']}`")

        auto_dish_toggle = st.checkbox("🤖 啟用 AI 自動辨識餐點類型 (Auto-Detect Dish)", value=True)
        candidate_dishes = df_dishes["name"].tolist()

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

        # 餐點名稱自動識別或手動覆蓋
        selected_dish = None
        if captured_image:
            if auto_dish_toggle:
                with st.spinner("AI 正在識別餐點品項 (CLIP Zero-Shot)..."):
                    detected_dish, dish_conf = predict_dish_category(captured_image, candidate_dishes, engine)
                st.success(f"🔍 **AI 自動辨識餐點**：`{detected_dish}` (置信度: {dish_conf:.1%})")
                selected_dish = detected_dish
            else:
                selected_dish = st.selectbox("抽檢餐點 (手動微調)", candidate_dishes)

        # 執行推論與寫入 DB
        if captured_image and should_run_detection:
            with st.spinner("Pipeline 1 & 2 運算中: YOLOS 繪製框線與 Flan-T5 生成決策..."):
                annotated_img, item_list, waste_ratio, primary_cat = run_tray_waste_detection(captured_image, engine)
                kitchen_advice = generate_portioning_directive(
                    selected_branch_name, b_row["level"], b_row["base_rice_g"], 
                    b_row["strategy"], selected_dish, waste_ratio, engine
                )

            saved_hkd, saved_co2 = calculate_cost_and_esg(waste_ratio, b_row["level"])
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            save_audit_record(db_record)

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
            st.toast("✅ 偵測完成！結果已自動同步至資料庫與總部大盤。")

    # 右側：最新偵測看板
    with col_right:
        st.markdown("### 🎯 最新偵測結果 (Latest Detection Result)")
        latest = st.session_state.get("latest_result")

        if latest is None:
            st.info("💡 尚未執行偵測。請保持 Live Camera 開啟並放置餐盤，或手動拍攝/上傳。")
            placeholder_img = Image.new("RGB", (400, 300), color=(240, 240, 240))
            d = ImageDraw.Draw(placeholder_img)
            d.text((120, 140), "等待餐盤輸入中...", fill=(150, 150, 150))
            st.image(placeholder_img, caption="即時預覽看板", use_container_width=True)
        else:
            st.image(latest["image"], caption=f"最新審計影像 [{latest['dish_name']}] ({latest['timestamp']})", use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("殘食佔比", f"{latest['waste_ratio']:.1%}")
            m2.metric("辨識菜品", latest["dish_name"].split(" ")[0])
            m3.metric("損耗金額", f"HK$ {latest['cost']}")
            st.success(f"👨‍🍳 **後廚出餐校準指令 ({latest['branch_name']})**:\n\n{latest['kitchen_advice']}")
            if latest["item_list"]:
                with st.expander("查看 Bounding Box 偵測物件明細", expanded=False):
                    st.dataframe(pd.DataFrame(latest["item_list"]), use_container_width=True)


def render_mode_2_dashboard():
    """Mode 2: 總部即時營運與 ESG 大盤視圖"""
    st.subheader("📊 Mode 2: 大家樂集團總部 - 跨分店即時營運與 ESG 大盤")
    st.caption("數據來源：直接讀取 SQLite 資料庫中來自 Mode 1 的真實偵測記錄")
    
    df_db = get_all_audit_records()
    if df_db.empty:
        st.warning("⚠️ 目前資料庫中無任何審計數據。請切換至「Mode 1」完成幾次掃描測試！")
        return

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


def render_mode_3_master_data(df_branches: pd.DataFrame, df_dishes: pd.DataFrame):
    """Mode 3: 基礎資料設定與 CSV 批量維護"""
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


# ==============================================================================
# 4. 主執行程序入口 (Main Execution Entry)
# ==============================================================================
def main():
    """主程序控制中心"""
    # 1. 初始化資料庫與全域資料
    init_sqlite_db()
    df_branches, df_dishes = load_master_meta()
    
    # 2. 載入 AI 模型引擎
    with st.spinner("🚀 正在啟動多模態 AI 引擎 (YOLOS-tiny + Flan-T5 + CLIP)..."):
        engine = init_ai_pipeline_engine()
        
    # 3. 渲染主頁面標題與導航
    render_header()
    selected_mode = render_sidebar_controls()
    
    # 4. 根據模式路由
    if selected_mode == "Mode 1: 前線餐盤智能偵測 (Tray Detection)":
        render_mode_1_detection(df_branches, df_dishes, engine)
    elif selected_mode == "Mode 2: 總部即時營運大盤 (Real-time Dashboard)":
        render_mode_2_dashboard()
    else:
        render_mode_3_master_data(df_branches, df_dishes)


if __name__ == "__main__":
    main()
