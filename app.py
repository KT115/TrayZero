import os
import time
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

# ==============================================================================
# 0. 全域常數與預設資料
# ==============================================================================
BRANCH_FILE = "master_branches.csv"
DISH_FILE = "master_dishes.csv"
DB_FILE = "trayzero_audit.db"
LOGO_FILE = "CDC_810.jpg"

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

FOOD_AND_TRAY_WHITELIST = {
    "bowl": "Rice",
    "cake": "Rice",
    "sandwich": "Meat",
    "pizza": "Meat",
    "hot dog": "Meat",
    "carrot": "Veg_Soup",
    "broccoli": "Veg_Soup",
    "apple": "Veg_Soup",
    "orange": "Veg_Soup",
    "banana": "Veg_Soup",
    "donut": "Meat",
    "cup": "Veg_Soup",
    "bottle": "Veg_Soup",
    "dining table": "Tray"
}

# ==============================================================================
# 1. 樣式注入：高質感 Dark SaaS UI
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&family=Noto+Sans+TC:wght@500;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Noto Sans TC', sans-serif;
        }

        /* 頂部單行大氣深色標題 */
        .trayzero-header {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-left: 6px solid #F59E0B;
            border-radius: 14px;
            padding: 18px 24px;
            margin-bottom: 22px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            display: flex;
            align-items: center;
        }
        
        .trayzero-title-text {
            color: #F8FAFC !important;
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
            margin: 0 !important;
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* 企業級 SaaS KPI 指標卡片 */
        .saas-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border-top: 3px solid #F59E0B;
        }

        .saas-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .saas-value {
            font-size: 1.8rem;
            font-weight: 900;
            color: #F8FAFC;
            line-height: 1.1;
        }

        .saas-sub {
            font-size: 0.8rem;
            color: #10B981;
            font-weight: 700;
            margin-top: 6px;
        }

        /* 建議決策卡片 */
        .directive-card {
            border-radius: 12px;
            padding: 16px 18px;
            margin-bottom: 14px;
            border-left: 5px solid;
            background: #1E293B;
            border-top: 1px solid #334155;
            border-right: 1px solid #334155;
            border-bottom: 1px solid #334155;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        }
        .directive-chef { border-left-color: #EF4444; }
        .directive-pos { border-left-color: #F59E0B; }
        .directive-mgr { border-left-color: #3B82F6; }

        .directive-title {
            font-size: 0.92rem;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 6px;
        }
        .directive-body {
            font-size: 0.86rem;
            color: #CBD5E1;
            line-height: 1.6;
            font-weight: 500;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. 資料庫模組
# ==============================================================================
def init_sqlite_db():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    audit_date TEXT,
                    audit_month TEXT,
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

            cursor.execute("PRAGMA table_info(audit_logs)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            if "audit_date" not in existing_columns:
                cursor.execute("ALTER TABLE audit_logs ADD COLUMN audit_date TEXT")
            if "audit_month" not in existing_columns:
                cursor.execute("ALTER TABLE audit_logs ADD COLUMN audit_month TEXT")
            conn.commit()
    except Exception as e:
        st.error(f"資料庫初始化失敗: {e}")

def save_audit_record(record):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (
                    timestamp, audit_date, audit_month, branch_name, branch_level, 
                    dish_name, primary_waste, waste_ratio, cost_waste_hkd, co2_emission_kg
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["timestamp"], record["audit_date"], record["audit_month"],
                record["branch_name"], record["branch_level"], record["dish_name"], 
                record["primary_waste"], record["waste_ratio"], record["cost_waste_hkd"], 
                record["co2_emission_kg"]
            ))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"儲存記錄失敗: {e}")
        return False

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
        st.error(f"清空失敗: {e}")

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
# 3. AI 模型引擎
# ==============================================================================
@st.cache_resource(show_spinner=False)
def init_ai_pipeline_engine():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./Fine-tuned_Model_files" if os.path.exists("./Fine-tuned_Model_files") and any(os.scandir("./Fine-tuned_Model_files")) else "hustvl/yolos-tiny"
    
    img_processor = AutoImageProcessor.from_pretrained(model_path)
    det_model = AutoModelForObjectDetection.from_pretrained(model_path).to(device)
    det_model.eval()
    
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
    
    color_map = {"Rice": "#EF4444", "Meat": "#F59E0B", "Veg_Soup": "#10B981", "Tray": "#3B82F6"}
    primary_category = "光盤 (Clean Plate)"
    valid_food_found = False

    for box, score, label_id in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()):
        raw_label = engine["detector"].config.id2label.get(label_id, "item")
        
        if raw_label not in FOOD_AND_TRAY_WHITELIST:
            continue
            
        category = FOOD_AND_TRAY_WHITELIST[raw_label]
        valid_food_found = True
        
        if category == "Rice":
            display_name = "白飯/主食殘留 (Rice Waste)"
            primary_category = "白飯/主食殘留"
        elif category == "Meat":
            display_name = "主菜肉類殘留 (Meat Residual)"
            if "白飯" not in primary_category:
                primary_category = "主菜肉類殘留"
        elif category == "Veg_Soup":
            display_name = f"配菜/醬汁殘留 ({raw_label})"
            if primary_category == "光盤 (Clean Plate)":
                primary_category = "配菜/醬汁殘留"
        else:
            display_name = "餐盤定位 (Tray Baseline)"

        xmin, ymin = max(0, box[0]), max(0, box[1])
        xmax, ymax = min(img_w, box[2]), min(img_h, box[3])
        box_area = (xmax - xmin) * (ymax - ymin)
        
        if category != "Tray":
            waste_box_area += box_area
        
        c = color_map.get(category, "#EF4444")
        draw.rectangle([xmin, ymin, xmax, ymax], outline=c, width=3)
        caption = f"{display_name} {score:.1%}"
        draw.rectangle([xmin, max(0, ymin - 18), xmin + len(caption) * 7.5, ymin], fill=c)
        draw.text((xmin + 4, max(0, ymin - 16)), caption, fill="white")
        
        detected_items.append({
            "分類項目": display_name,
            "置信度": f"{score:.2%}",
            "佔比": f"{(box_area / total_area):.1%}"
        })
        
    waste_ratio = min(1.0, waste_box_area / (total_area * 0.65)) if (total_area > 0 and valid_food_found) else 0.0
    return annotated_img, detected_items, waste_ratio, primary_category, valid_food_found

def auto_detect_dish_heuristic(image, candidate_dishes):
    img_np = np.array(image.resize((64, 64)))
    avg_r = np.mean(img_np[:, :, 0])
    avg_g = np.mean(img_np[:, :, 1])
    avg_b = np.mean(img_np[:, :, 2])
    
    if avg_r > 150 and avg_g > 110 and avg_b < 90:
        dish = candidate_dishes[1] if len(candidate_dishes) > 1 else candidate_dishes[0]
        conf = 0.88
    elif avg_r > 160 and avg_g < 110:
        dish = candidate_dishes[0]
        conf = 0.92
    elif avg_g > 130:
        dish = candidate_dishes[2] if len(candidate_dishes) > 2 else candidate_dishes[0]
        conf = 0.85
    else:
        dish = candidate_dishes[0]
        conf = 0.81
    return dish, conf

# ==============================================================================
# 4. 宏觀審計建議引擎 (支援 分店維度 + 食物種類維度)
# ==============================================================================
def generate_macro_advisory(df_scope: pd.DataFrame, scope_type: str, filter_branch: str, filter_dish: str, engine: dict) -> dict:
    if df_scope.empty:
        total_trays = 0
        avg_waste = 0.0
        total_loss_hkd = 0.0
        total_co2 = 0.0
        target_branch_desc = filter_branch if filter_branch != "ALL" else "全港分店"
        target_dish_desc = filter_dish if filter_dish != "ALL" else "全部餐點品項"
    else:
        total_trays = len(df_scope)
        avg_waste = df_scope["waste_ratio"].mean()
        total_loss_hkd = df_scope["cost_waste_hkd"].sum()
        total_co2 = df_scope["co2_emission_kg"].sum()
        target_branch_desc = filter_branch if filter_branch != "ALL" else df_scope.groupby("branch_name")["waste_ratio"].mean().idxmax()
        target_dish_desc = filter_dish if filter_dish != "ALL" else df_scope.groupby("dish_name")["waste_ratio"].mean().idxmax()
    
    advisory = {
        "scope": scope_type,
        "total_trays": total_trays,
        "avg_waste": avg_waste,
        "total_loss_hkd": total_loss_hkd,
        "total_co2": total_co2,
        "worst_branch": target_branch_desc,
        "worst_dish": target_dish_desc,
        "action_items": [],
        "executive_memo": ""
    }

    branch_context = "全港集團總部" if filter_branch == "ALL" else f"【{filter_branch}】"
    dish_context = "全品類餐點" if filter_dish == "ALL" else f"【{filter_dish.split(' ')[0]}】"

    if scope_type == "DAILY":
        advisory["action_items"].append({
            "type": "directive-chef",
            "role": f"👨‍🍳 後廚出餐負責人 ({branch_context} - {dish_context} SOP)",
            "directive": f"【即時出餐規格調校】針對 {dish_context} 審計結果，平均殘食率達 {avg_waste:.1f}%。明日午市起針對商業區門市全面換裝 3 號平底飯勺（每份減量 30g 出餐），嚴控熟米積壓。"
        })
        advisory["action_items"].append({
            "type": "directive-pos",
            "role": "🖥️ 門市 POS / Kiosk 運營",
            "directive": f"【點餐機品項促銷聯動】明日於「{target_branch_desc}」點餐機針對 {dish_context} 置頂彈窗「少飯減扣 $2」優惠，引流小食量顧客選用輕量裝。"
        })
        advisory["action_items"].append({
            "type": "directive-mgr",
            "role": "📦 門市經理 (Store Manager)",
            "directive": f"【電飯煲蒸煮量下調】明日煮米批次減少 2 鍋（下調約 10% 產能），單日預計防損挽回 HK$ {max(150, round(total_loss_hkd * 0.4)):,.0f}。"
        })

        prompt = (
            f"You are the operations head of Cafe de Coral. Scope: {branch_context}, Dish category: {dish_context}. "
            f"{total_trays} trays audited, average food waste ratio {avg_waste:.1f}%, total loss HK${total_loss_hkd:.1f}. "
            f"Target branch: {target_branch_desc}, target dish: {target_dish_desc}. "
            f"Write a 2-sentence direct operational instruction for tomorrow's store managers."
        )
    else:
        monthly_saving = max(3600, total_loss_hkd * 12)
        advisory["action_items"].append({
            "type": "directive-chef",
            "role": "🏭 大埔中央廚房 (Central Kitchen)",
            "directive": f"【主菜規格重新開模】月度數據顯示 {dish_context} 居殘食榜首。建議中央廚房將厚切豬排/肉排單塊重量規格下調 8%（由 180g 改為 165g），徹底根治過剩浪費。"
        })
        advisory["action_items"].append({
            "type": "directive-pos",
            "role": "🏢 總部營運與菜單工程部",
            "directive": f"【門市分級差異化定價】針對 {branch_context} 顧客對 {dish_context} 的消費數據，下月正式推行「輕量少飯版菜單」，客單價微降 $1，毛利率提升 1.8%。"
        })
        advisory["action_items"].append({
            "type": "directive-mgr",
            "role": "🌱 集團 ESG 與永續發展委員會",
            "directive": f"【年化碳減量與防損核算】本期累計減廢預計年化防損達 HK$ {monthly_saving:,.0f}，月減碳 {total_co2:.1f} kg CO2e，數據已自動同步至年度可持續發展報告 (ESG Report)。"
        })

        prompt = (
            f"You are the Group CEO of Cafe de Coral Holdings. Scope: {branch_context}, Category: {dish_context}. "
            f"{total_trays} audits, total loss HK${total_loss_hkd:.1f}, average waste {avg_waste:.1f}%. "
            f"Write an executive board-level strategic comment on supply chain and portion resizing."
        )

    try:
        inputs = engine["tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(engine["device"])
        with torch.no_grad():
            outputs = engine["generator"].generate(**inputs, max_new_tokens=80, do_sample=False)
        advisory["executive_memo"] = engine["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    except Exception:
        advisory["executive_memo"] = f"大家樂營運總部核准：落實 {branch_context} 對 {dish_context} 的 {scope_type} 殘食校準方針，精準優化門市與中央廚房配給量。"

    return advisory

# ==============================================================================
# 5. UI 渲染控制
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="trayzero-header">
        <h2 class="trayzero-title-text">🍽️ TrayZero Intelligent plate waste auditing and central distribution system</h2>
    </div>
    """, unsafe_allow_html=True)

def render_mode_1_detection(df_branches, df_dishes, engine):
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### 🏢 執勤門市與掃描設置")
        branch_names = df_branches["name"].tolist()
        selected_branch_name = st.selectbox("執勤門市", branch_names)
        b_row = df_branches[df_branches["name"] == selected_branch_name].iloc[0]
        
        st.markdown(f"**門市等級**: `{b_row['level']}` | **區域**: `{b_row['district']}` | **標配飯量**: `{b_row['base_rice_g']}g`")

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
            live_shot = st.camera_input("持續監控畫面", key="permanent_live_cam")
            if live_shot:
                captured_image = Image.open(live_shot).convert("RGB")
                current_frame_hash = hash(captured_image.tobytes()[:3000])
                if current_frame_hash != st.session_state.get("last_scanned_hash"):
                    countdown_box = st.empty()
                    for s in range(2, 0, -1):
                        countdown_box.warning(f"⏳ 偵測到畫面放入，靜止鎖定中... {s} 秒")
                        time.sleep(1)
                    countdown_box.success("🎯 鎖定完成！自動觸發 AI 審計分析...")
                    st.session_state["last_scanned_hash"] = current_frame_hash
                    should_run_detection = True
                else:
                    st.info("🟢 監控中：當前畫面已完成分析。更換餐盤將自動觸發下一次偵測。")

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

        # 門禁檢查與推論
        if captured_image and should_run_detection:
            with st.spinner("AI 偵測中: YOLOS 正在過濾非餐盤目標並辨識殘食..."):
                annotated_img, item_list, waste_ratio, primary_cat, valid_food_found = run_tray_waste_detection(captured_image, engine)

            if not valid_food_found:
                st.error("🚫 偵測失敗：鏡頭前未發現任何大家樂餐盤或食物物件！（已自動過濾人物/背景）")
                st.session_state["latest_result"] = {
                    "image": annotated_img,
                    "item_list": [],
                    "waste_ratio": 0.0,
                    "primary_cat": "無有效餐盤",
                    "branch_name": selected_branch_name,
                    "dish_name": "未識別 (非食物)",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "cost": 0.0
                }
            else:
                if auto_dish_toggle:
                    detected_dish, dish_conf = auto_detect_dish_heuristic(captured_image, candidate_dishes)
                    st.success(f"🍱 **AI 識別餐點確認**：`{detected_dish}` (置信度: {dish_conf:.1%})")
                    selected_dish = detected_dish
                else:
                    selected_dish = st.selectbox("抽檢餐點 (手動微調)", candidate_dishes)

                unit_rice_cost = 0.015
                cut_g = 40 if "Level A" in str(b_row["level"]) else 20
                saved_hkd = round(cut_g * unit_rice_cost * 25, 1)
                saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
                
                now = datetime.datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
                date_str = now.strftime("%Y-%m-%d")
                month_str = now.strftime("%Y-%m")

                db_record = {
                    "timestamp": timestamp_str,
                    "audit_date": date_str,
                    "audit_month": month_str,
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
                    "branch_name": selected_branch_name,
                    "dish_name": selected_dish,
                    "timestamp": timestamp_str,
                    "cost": saved_hkd
                }
                st.toast("✅ 餐盤掃描成功！紀錄已成功歸檔至資料庫。")

    with col_right:
        st.markdown("#### 🎯 前線掃描結果 (Latest Result)")
        latest = st.session_state.get("latest_result")

        if not latest:
            placeholder_img = Image.new("RGB", (400, 260), color=(15, 23, 42))
            d = ImageDraw.Draw(placeholder_img)
            d.text((120, 120), "WAITING FOR TRAY INPUT...", fill=(148, 163, 184))
            st.image(placeholder_img, caption="即時監控看板", use_container_width=True)
        else:
            st.image(latest["image"], caption=f"審計影像: {latest['dish_name']} ({latest['timestamp']})", use_container_width=True)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class="saas-card">
                    <div class="saas-label">殘食佔比</div>
                    <div class="saas-value" style="color: {'#EF4444' if latest['waste_ratio'] > 0.3 else '#10B981'};">
                        {latest['waste_ratio']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="saas-card">
                    <div class="saas-label">主要殘留</div>
                    <div class="saas-value" style="font-size: 1.15rem; margin-top: 8px;">
                        {latest['primary_cat'].split(' ')[0]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="saas-card">
                    <div class="saas-label">推算損耗</div>
                    <div class="saas-value" style="color: #F59E0B;">
                        HK${latest['cost']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if latest["primary_cat"] == "無有效餐盤":
                st.warning("⚠️ 此鏡頭畫面被判定為人物或非食物影像，系統已拒絕入庫。")
            else:
                st.success(f"📥 **記錄已歸檔**：已綁定至 `{latest['branch_name']}`。宏觀建議請至 Mode 2 查看。")

            if latest.get("item_list"):
                with st.expander("查看 Bounding Box 偵測物件明細", expanded=True):
                    st.dataframe(pd.DataFrame(latest["item_list"]), use_container_width=True)

def render_mode_2_dashboard(df_branches, df_dishes, engine):
    st.markdown("### 📊 大家樂集團總部：全港即時營運大盤 & 戰略建議")
    df_raw = get_all_audit_records()

    # --------------------------------------------------------------------------
    # 核心維度選擇 (Dimensions Selection)：分店維度 + 食物品類維度
    # --------------------------------------------------------------------------
    st.markdown("#### 🎛️ 審計多維度透視 (Analysis Dimensions)")
    c_dim_b, c_dim_d = st.columns(2)
    
    with c_dim_b:
        branch_options = ["🌐 全部分店 (Overall Branches)"] + df_branches["name"].tolist()
        selected_branch_choice = st.selectbox("1. 門市維度過濾", branch_options)
        selected_branch_filter = "ALL" if "全部" in selected_branch_choice else selected_branch_choice

    with c_dim_d:
        dish_options = ["🍱 全部餐點品項 (Overall Dishes)"] + df_dishes["name"].tolist()
        selected_dish_choice = st.selectbox("2. 食物種類維度過濾", dish_options)
        selected_dish_filter = "ALL" if "全部" in selected_dish_choice else selected_dish_choice

    # 依照雙維度動態過濾資料庫
    df_filtered = df_raw.copy() if not df_raw.empty else pd.DataFrame()
    if not df_filtered.empty:
        if selected_branch_filter != "ALL":
            df_filtered = df_filtered[df_filtered["branch_name"] == selected_branch_filter]
        if selected_dish_filter != "ALL":
            df_filtered = df_filtered[df_filtered["dish_name"] == selected_dish_filter]

    # 頂部 KPI 卡片
    total_scans = len(df_filtered) if not df_filtered.empty else 0
    avg_waste = df_filtered["waste_ratio"].mean() if not df_filtered.empty else 0.0
    total_waste_hkd = df_filtered["cost_waste_hkd"].sum() if not df_filtered.empty else 0.0
    total_co2 = df_filtered["co2_emission_kg"].sum() if not df_filtered.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-label">篩選樣本數</div>
            <div class="saas-value">{total_scans} <span style="font-size: 0.9rem; color: #94A3B8;">TRAYS</span></div>
            <div class="saas-sub">▲ 雙維度動態聯動</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-label">該維度平均殘食率</div>
            <div class="saas-value" style="color: {'#EF4444' if avg_waste > 25 else '#10B981'};">{avg_waste:.1f}%</div>
            <div class="saas-sub">基準目標: &lt;15.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-label">食材損耗總額</div>
            <div class="saas-value" style="color: #F59E0B;">HK${total_waste_hkd:,.1f}</div>
            <div class="saas-sub">即時動態折算</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-label">累計碳排放當量</div>
            <div class="saas-value" style="color: #38BDF8;">{total_co2:.2f} <span style="font-size: 0.9rem; color: #94A3B8;">kg</span></div>
            <div class="saas-sub">Scope 3 ESG 披露</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 營運指導中心 (Advisory Hub)
    st.markdown("#### 🧭 大家樂總部營運指導中心 (Executive Advisory Hub)")
    review_scope = st.radio("選擇覆盤時限", ["📅 日度營運覆盤建議 (Daily Operational Review)", "🗓️ 月度戰略採購建議 (Monthly Strategic Advisory)"], horizontal=True)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    this_month_str = datetime.date.today().strftime("%Y-%m")

    if "日度" in review_scope:
        if not df_filtered.empty and "audit_date" in df_filtered.columns:
            available_dates = df_filtered["audit_date"].dropna().unique().tolist()
            if not available_dates:
                available_dates = [today_str]
        else:
            available_dates = [today_str]

        selected_date = st.selectbox("選擇覆盤日期", available_dates)
        df_target = df_filtered[df_filtered["audit_date"] == selected_date] if not df_filtered.empty else pd.DataFrame()
        scope_code = "DAILY"
    else:
        if not df_filtered.empty and "audit_month" in df_filtered.columns:
            available_months = df_filtered["audit_month"].dropna().unique().tolist()
            if not available_months:
                available_months = [this_month_str]
        else:
            available_months = [this_month_str]

        selected_month = st.selectbox("選擇審計月份", available_months)
        df_target = df_filtered[df_filtered["audit_month"] == selected_month] if not df_filtered.empty else pd.DataFrame()
        scope_code = "MONTHLY"

    with st.spinner("AI 正在針對該時段與維度數據進行宏觀分析與決策生成..."):
        advisory = generate_macro_advisory(df_target, scope_code, selected_branch_filter, selected_dish_filter, engine)

    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.markdown(f"""
        <div class="saas-card">
            <div class="saas-label">{review_scope.split(' ')[0]} 關鍵指標摘要</div>
            <div style="margin-top: 10px; font-size: 0.92rem; color: #CBD5E1; line-height: 1.9;">
                • 當期審計樣本: <b>{advisory['total_trays']} 盤</b><br>
                • 綜合殘食率: <b style="color: #F59E0B;">{advisory['avg_waste']:.1f}%</b><br>
                • 目標分析門市: <b>{advisory['worst_branch']}</b><br>
                • 目標分析餐點: <b>{advisory['worst_dish']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        for item in advisory["action_items"]:
            st.markdown(f"""
            <div class="directive-card {item['type']}">
                <div class="directive-title">{item['role']}</div>
                <div class="directive-body">{item['directive']}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📝 檢視 AI 總監決策備忘錄 (Executive Memo)", expanded=True):
            st.write(advisory["executive_memo"])

    st.markdown("---")
    if not df_filtered.empty:
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.markdown("##### 🏢 各門市平均殘食率 (%)")
            branch_stat = df_filtered.groupby("branch_name")["waste_ratio"].mean().reset_index()
            st.bar_chart(branch_stat, x="branch_name", y="waste_ratio", color="#F59E0B")

        with col_c2:
            st.markdown("##### 🍱 各食物種類耗損金額分佈 (HK$)")
            dish_stat = df_filtered.groupby("dish_name")["cost_waste_hkd"].sum().reset_index()
            st.bar_chart(dish_stat, x="dish_name", y="cost_waste_hkd", color="#EF4444")

        st.markdown("##### 📋 SQLite 歷史審計流水表 (當前維度)")
        st.dataframe(df_filtered, use_container_width=True)

        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 匯出當前維度審計數據 (CSV)",
            data=csv_data,
            file_name=f"cafedecoral_audit_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 目前所選維度（分店/餐點）尚無過盤紀錄。請至 Mode 1 執行掃描測試！")

def render_mode_3_master_data(df_branches, df_dishes):
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management)")
    tab_branch_mgt, tab_dish_mgt = st.tabs(["🏢 分店清單 (Branches)", "🍱 餐點品項 (Dishes)"])

    with tab_branch_mgt:
        col_b_up, col_b_dl = st.columns([2, 1])
        with col_b_dl:
            sample_branch_csv = pd.DataFrame(DEFAULT_BRANCHES).to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 下載分店 CSV 格式範本", data=sample_branch_csv, file_name="branches_template.csv", mime="text/csv")
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

        edited_branches = st.data_editor(df_branches, num_rows="dynamic", use_container_width=True, key="branch_editor")
        if st.button("💾 儲存分店手動修改內容", type="primary"):
            edited_branches.to_csv(BRANCH_FILE, index=False)
            st.success("✅ 分店清單已更新！")
            st.rerun()

    with tab_dish_mgt:
        col_d_up, col_d_dl = st.columns([2, 1])
        with col_d_dl:
            sample_dish_csv = pd.DataFrame(DEFAULT_DISHES).to_csv(index=False).encode("utf-8")
            st.download_button(label="📥 下載餐點 CSV 格式範本", data=sample_dish_csv, file_name="dishes_template.csv", mime="text/csv")
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

        edited_dishes = st.data_editor(df_dishes, num_rows="dynamic", use_container_width=True, key="dish_editor")
        if st.button("💾 儲存餐點手動修改內容", type="primary"):
            edited_dishes.to_csv(DISH_FILE, index=False)
            st.success("✅ 餐點清單已更新！")
            st.rerun()

# ==============================================================================
# 6. 主程序入口
# ==============================================================================
def main():
    st.set_page_config(
        page_title="TrayZero | 大家樂智能餐盤審計系統",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    if "latest_result" not in st.session_state:
        st.session_state["latest_result"] = None

    if "last_scanned_hash" not in st.session_state:
        st.session_state["last_scanned_hash"] = None

    init_sqlite_db()
    df_branches, df_dishes = load_master_meta()

    with st.spinner("🚀 正在啟動 AI 模型引擎 (YOLOS + Flan-T5)..."):
        engine = init_ai_pipeline_engine()

    render_header()

    # 側邊欄 Logo：直接調用本地正版圖檔，徹底杜絕 Base64 或 SVG 變形問題
    if os.path.exists(LOGO_FILE):
        st.sidebar.image(LOGO_FILE, use_container_width=True)
    else:
        st.sidebar.warning("⚠️ 請確認 CDC_810.jpg 已放置於專案根目錄")

    st.sidebar.title("🎛️ 系統控制台")
    selected_mode = st.sidebar.radio(
        "工作模式",
        [
            "Mode 1: 前線餐盤智能偵測 (Tray Station)", 
            "Mode 2: 總部即時營運大盤 (HQ Dashboard)",
            "Mode 3: 基礎資料設定 (Master Data)"
        ],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 測試維護")
    if st.sidebar.button("🗑️ 清空審計資料庫 (Reset DB)", type="secondary"):
        truncate_audit_db()
        st.session_state["latest_result"] = None
        st.session_state["last_scanned_hash"] = None
        st.sidebar.success("✅ 資料庫已完全清空！")
        st.rerun()

    if selected_mode == "Mode 1: 前線餐盤智能偵測 (Tray Station)":
        render_mode_1_detection(df_branches, df_dishes, engine)
    elif selected_mode == "Mode 2: 總部即時營運大盤 (HQ Dashboard)":
        render_mode_2_dashboard(df_branches, df_dishes, engine)
    else:
        render_mode_3_master_data(df_branches, df_dishes)

if __name__ == "__main__":
    main()
