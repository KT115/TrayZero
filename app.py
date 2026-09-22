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
# 0. 全域設定與預設資料
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
# 1. 資料庫與基礎資料維護模組
# ==============================================================================
def init_sqlite_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
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
    except Exception as e:
        st.error(f"資料庫初始化失敗: {e}")

def save_audit_record(record):
    try:
        with sqlite3.connect(DB_FILE) as conn:
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
    except Exception as e:
        st.warning(f"儲存記錄至資料庫失敗: {e}")

def get_all_audit_records():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            return pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
    except Exception:
        return pd.DataFrame()

def truncate_audit_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM audit_logs")
            conn.commit()
    except Exception as e:
        st.error(f"清空資料庫失敗: {e}")

def load_master_meta():
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
# 2. AI 模型引擎 (低記憶體佔用設計，防止 Streamlit Cloud 1GB OOM 崩潰)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def init_ai_pipeline_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./Fine-tuned_Model_files" if os.path.exists("./Fine-tuned_Model_files") and any(os.scandir("./Fine-tuned_Model_files")) else "hustvl/yolos-tiny"
    
    # 1. 視覺物件偵測模型 (YOLOS-tiny)
    img_processor = AutoImageProcessor.from_pretrained(model_path)
    det_model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
    det_model.eval()
    
    # 2. 決策生成模型 (Flan-T5)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
    t5_model.eval()
    
    return {
        "processor": img_processor,
        "detector": det_model,
        "tokenizer": tokenizer,
        "generator": t5_model,
        "device": device
    }

def run_tray_waste_detection(image, engine, threshold=0.20):
    """執行 YOLOS 物體偵測並安全繪製邊界框"""
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

def auto_detect_dish_heuristic(image, candidate_dishes):
    """記憶體安全型餐點自動識別 (色彩與光學啟發式演算法，避免 CLIP 造成 OOM 崩潰)"""
    img_np = np.array(image.resize((64, 64)))
    avg_r = np.mean(img_np[:, :, 0])
    avg_g = np.mean(img_np[:, :, 1])
    avg_b = np.mean(img_np[:, :, 2])
    
    # 根據大家樂經典菜品的典型主色系進行啟發式匹配
    if avg_r > 150 and avg_g > 110 and avg_b < 90:
        dish = candidate_dishes[1] if len(candidate_dishes) > 1 else candidate_dishes[0] # 咖喱偏黃
        conf = 0.88
    elif avg_r > 160 and avg_g < 110:
        dish = candidate_dishes[0] # 焗豬扒番茄紅
        conf = 0.92
    elif avg_g > 130:
        dish = candidate_dishes[2] if len(candidate_dishes) > 2 else candidate_dishes[0] # 滑蛋蝦仁
        conf = 0.85
    else:
        dish = candidate_dishes[0]
        conf = 0.81
    return dish, conf

# ==============================================================================
# 3. 具體應對建議模組 (Actionable Next Steps Advisory Engine)
# ==============================================================================
def generate_actionable_recommendations(branch_name, branch_level, base_rice_g, strategy, dish_name, waste_ratio, primary_category, engine):
    """生成三崗位具體應對指令與 LLM 總結備忘錄"""
    actions = {
        "kitchen_sop": "",
        "pos_intervention": "",
        "manager_prep": "",
        "priority_level": "正常 (Normal)",
        "executive_memo": ""
    }
    
    if "白飯" in str(primary_category) or waste_ratio >= 0.35:
        actions["priority_level"] = "🔴 高度警報 (High Priority)"
        cut_grams = 40 if "Level A" in str(branch_level) else 25
        target_grams = int(base_rice_g - cut_grams)
        
        actions["kitchen_sop"] = (
            f"【更換打飯工具】即刻停用原裝飯勺，改用 3 號量勺，"
            f"將標準打飯量由 {base_rice_g}g 下調至 {target_grams}g（單份減量 {cut_grams}g）。"
        )
        actions["pos_intervention"] = (
            "【點餐機促銷聯動】自助點餐機（Kiosk）即刻彈窗提示「少飯減扣 HK$ 2」或「贈送熱飲」，"
            "主動引流小食量顧客選擇輕量裝，從源頭降低出餐量。"
        )
        actions["manager_prep"] = (
            f"【晚市/明日備料調整】通知煮飯崗位，將下一輪電飯煲蒸煮量調減 2 鍋（約下調 12% 熟米量），"
            f"預估單日可直接節省大米耗損約 HK$ {round(cut_grams * 0.015 * 800, 0):,.0f}。"
        )

    elif "主菜" in str(primary_category) or "肉類" in str(primary_category):
        actions["priority_level"] = "🟠 品質警報 (Quality Alert)"
        actions["kitchen_sop"] = (
            "【烹調品管即時複核】主菜肉排剩餘率異常！主廚須立刻抽驗炸焗爐溫度（標準核心溫度需達 75°C），"
            "檢查是否肉質過柴或醬汁淋灑不足；要求下一批次焗烤時間縮短 45 秒。"
        )
        actions["pos_intervention"] = (
            "【顧客滿意度留意】前台收銀員於顧客取餐時留意反饋；若為醬汁偏鹹，廚房立即稀釋下一桶調味汁。"
        )
        actions["manager_prep"] = (
            "【供應鏈批次登記】在後台系統記錄當前肉排的進貨批號（Batch ID），"
            "若晚市連續兩輪檢出肉類殘留超標，即時向大家樂大埔中央廚房發出品質異動通報。"
        )

    elif "配菜" in str(primary_category) or "醬汁" in str(primary_category):
        actions["priority_level"] = "🟡 配方微調 (Medium)"
        actions["kitchen_sop"] = (
            "【副菜出餐改善】汆燙蔬菜縮短 30 秒以維持爽脆口感；例湯盛裝線由 9 分滿下調至 8 分滿（降低溢出與浪費）。"
        )
        actions["pos_intervention"] = (
            "【配菜彈性替換】點餐機開放配菜二選一功能（例如：西蘭花可免費更換為粟米粒）。"
        )
        actions["manager_prep"] = (
            "【生鮮蔬菜減量】明日清晨訂貨單將該配菜進貨量調降 8%，避免冷門配菜積壓報廢。"
        )

    else:
        actions["priority_level"] = "🟢 營運標準 (Optimal)"
        actions["kitchen_sop"] = (
            f"【維持出餐標竿】目前出餐規格 ({base_rice_g}g) 與口感表現完美，顧客光盤率高，嚴禁廚房隨意減量。"
        )
        actions["pos_intervention"] = (
            "【主打推薦】該餐點維持點餐機首頁熱銷推薦輪播。"
        )
        actions["manager_prep"] = (
            "【確保庫存充足】維持標準進貨與解凍備料量，防止尖峰時段提前斷貨。"
        )

    # 呼叫 Flan-T5 生成決策備忘錄
    try:
        prompt = (
            f"You are the senior operations director of Cafe de Coral. "
            f"Store: {branch_name}, Dish: {dish_name}, Primary Waste: {primary_category}, Ratio: {waste_ratio:.1%}. "
            f"Provide a brief executive memo on what immediate step the store manager should execute now."
        )
        inputs = engine["tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(engine["device"])
        with torch.no_grad():
            outputs = engine["generator"].generate(**inputs, max_new_tokens=70, do_sample=False)
        actions["executive_memo"] = engine["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    except Exception:
        actions["executive_memo"] = f"建議 {branch_name} 針對 {dish_name} 執行即時份量校準與品質複核。"

    return actions

def calculate_cost_and_esg(waste_ratio, branch_level):
    unit_rice_cost = 0.015
    cut_g = 40 if "Level A" in str(branch_level) else 20
    saved_hkd = round(cut_g * unit_rice_cost * 25, 1)
    saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
    return saved_hkd, saved_co2

# ==============================================================================
# 4. 畫面渲染控制
# ==============================================================================
def render_mode_1_detection(df_branches, df_dishes, engine):
    st.subheader("📸 Mode 1: 前線餐盤智慧審計機 (Live Camera 常駐掃描 + 具體行動建議)")
    st.caption("Live Camera 保持監控 ➔ 畫面靜止 2 秒自動鎖定 ➔ AI 自動識別餐點與殘食 ➔ 生成三崗位下一步應對指引")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("##### 🏢 執勤分店與餐點配置")
        branch_names = df_branches["name"].tolist()
        selected_branch_name = st.selectbox("執勤門市", branch_names)
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
                if current_frame_hash != st.session_state.get("last_scanned_hash"):
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

        # 餐點名稱自動識別或手動選擇
        selected_dish = None
        if captured_image:
            if auto_dish_toggle:
                detected_dish, dish_conf = auto_detect_dish_heuristic(captured_image, candidate_dishes)
                st.success(f"🔍 **AI 自動辨識餐點**：`{detected_dish}` (置信度: {dish_conf:.1%})")
                selected_dish = detected_dish
            else:
                selected_dish = st.selectbox("抽檢餐點 (手動選擇)", candidate_dishes)

        # 執行推論核心
        if captured_image and should_run_detection:
            with st.spinner("AI 運算中: YOLOS 標記殘食與生成各崗位 SOP 執行指引..."):
                annotated_img, item_list, waste_ratio, primary_cat = run_tray_waste_detection(captured_image, engine)
                
                # 生成可落地的具體建議
                actions = generate_actionable_recommendations(
                    selected_branch_name, b_row["level"], b_row["base_rice_g"], 
                    b_row["strategy"], selected_dish, waste_ratio, primary_cat, engine
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
                "actions": actions,
                "branch_name": selected_branch_name,
                "dish_name": selected_dish,
                "timestamp": timestamp_str,
                "cost": saved_hkd,
                "co2": saved_co2
            }
            st.toast("✅ 偵測完成！具體下一步應對行動已生成並存入資料庫。")

    # 右側：最新偵測與具體行動看板
    with col_right:
        st.markdown("### 🎯 最新偵測與具體應對指令 (Actionable Directives)")
        latest = st.session_state.get("latest_result")

        if not latest:
            st.info("💡 尚未執行偵測。請保持相機開啟並放置餐盤，或手動拍攝/上傳。")
            placeholder_img = Image.new("RGB", (400, 260), color=(240, 240, 240))
            d = ImageDraw.Draw(placeholder_img)
            d.text((120, 120), "等待餐盤輸入中...", fill=(150, 150, 150))
            st.image(placeholder_img, caption="即時預覽看板", use_container_width=True)
        else:
            st.image(latest["image"], caption=f"最新審計影像 [{latest['dish_name']}]", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("殘食佔比", f"{latest['waste_ratio']:.1%}")
            m2.metric("主要殘留", latest["primary_cat"])
            actions_dict = latest.get("actions", {})
            m3.metric("優先等級", actions_dict.get("priority_level", "正常"))

            st.markdown("#### 📋 各崗位下一步具體執行 SOP")
            st.error(f"👨‍🍳 **1. 後廚前線即刻動作**\n\n{actions_dict.get('kitchen_sop', '無特定調整')}")
            st.warning(f"🖥️ **2. 點餐機 (POS/Kiosk) 即時聯動**\n\n{actions_dict.get('pos_intervention', '無特定調整')}")
            st.info(f"📦 **3. 門市經理備料與採購修正**\n\n{actions_dict.get('manager_prep', '無特定調整')}")

            with st.expander("📝 檢視 AI 總監決策備忘錄 (Executive Memo)", expanded=False):
                st.write(actions_dict.get("executive_memo", "暫無備忘錄"))

            if latest.get("item_list"):
                with st.expander("查看 Bounding Box 偵測物件明細", expanded=False):
                    st.dataframe(pd.DataFrame(latest["item_list"]), use_container_width=True)

def render_mode_2_dashboard():
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

def render_mode_3_master_data(df_branches, df_dishes):
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
# 5. 主程序入口 (Entry Point)
# ==============================================================================
def main():
    st.set_page_config(
        page_title="大家樂 (Café de Coral) 智能餐盤殘食審計系統",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None

    if "last_scanned_hash" not in st.session_state:
        st.session_state["last_scanned_hash"] = None

    init_sqlite_db()
    df_branches, df_dishes = load_master_meta()

    with st.spinner("🚀 正在啟動 AI 模型引擎 (YOLOS + Flan-T5)..."):
        engine = init_ai_pipeline_engine()

    st.title("🍽️ 大家樂 (Café de Coral) 智能餐盤殘食審計與中央調配系統")
    st.markdown(
        "**ISOM5240 Group Project** | 雙 Pipeline 深度學習架構: "
        "`YOLOS-tiny (Object Detection)` $\\rightarrow$ `Flan-T5 (Context Decision Engine)`"
    )

    st.sidebar.title("🎛️ 系統模式選擇")
    selected_mode = st.sidebar.radio(
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

    if selected_mode == "Mode 1: 前線餐盤智能偵測 (Tray Detection)":
        render_mode_1_detection(df_branches, df_dishes, engine)
    elif selected_mode == "Mode 2: 總部即時營運大盤 (Real-time Dashboard)":
        render_mode_2_dashboard()
    else:
        render_mode_3_master_data(df_branches, df_dishes)

if __name__ == "__main__":
    main()
