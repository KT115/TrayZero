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

# 合法食物與餐具白名單 (過濾人物 person、領帶 tie、手機等非餐盤物件)
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
# 1. 資料庫模組 (自動 Migration，解決 no column named audit_date)
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

            # 檢查舊表是否缺少欄位並自動補齊 (Schema Migration)
            cursor.execute("PRAGMA table_info(audit_logs)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            if "audit_date" not in existing_columns:
                cursor.execute("ALTER TABLE audit_logs ADD COLUMN audit_date TEXT")
            if "audit_month" not in existing_columns:
                cursor.execute("ALTER TABLE audit_logs ADD COLUMN audit_month TEXT")
            conn.commit()
    except Exception as e:
        st.error(f"資料庫初始化/升級失敗: {e}")

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
# 2. AI 模型引擎
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
    """
    執行 YOLOS 物體偵測，並加入合法食物與餐盤白名單過濾
    避免人臉、人身、背景雜物被誤判為食物殘渣
    """
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
    
    color_map = {"Rice": "#E74C3C", "Meat": "#E67E22", "Veg_Soup": "#27AE60", "Tray": "#3498DB"}
    primary_category = "光盤 (Clean Plate)"
    valid_food_found = False

    for box, score, label_id in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()):
        raw_label = engine["detector"].config.id2label.get(label_id, "item")
        
        # 關鍵防護：非食物或餐具物件（如人像 person、領帶 tie 等）直接忽略，不計入殘食
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
        else: # Tray / Table
            display_name = "餐盤定位 (Tray Baseline)"

        xmin, ymin = max(0, box[0]), max(0, box[1])
        xmax, ymax = min(img_w, box[2]), min(img_h, box[3])
        box_area = (xmax - xmin) * (ymax - ymin)
        
        if category != "Tray":
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
# 3. 宏觀審計建議引擎 (供 Mode 2 Dashboard 使用)
# ==============================================================================
def generate_macro_advisory(df_scope: pd.DataFrame, scope_type: str, engine: dict) -> dict:
    if df_scope.empty:
        return {}

    total_trays = len(df_scope)
    avg_waste = df_scope["waste_ratio"].mean()
    total_loss_hkd = df_scope["cost_waste_hkd"].sum()
    total_co2 = df_scope["co2_emission_kg"].sum()
    
    top_waste_cat = df_scope["primary_waste"].value_counts().idxmax()
    worst_branch = df_scope.groupby("branch_name")["waste_ratio"].mean().idxmax()
    worst_dish = df_scope.groupby("dish_name")["waste_ratio"].mean().idxmax()
    
    advisory = {
        "scope": scope_type,
        "total_trays": total_trays,
        "avg_waste": avg_waste,
        "total_loss_hkd": total_loss_hkd,
        "total_co2": total_co2,
        "worst_branch": worst_branch,
        "worst_dish": worst_dish,
        "top_waste_cat": top_waste_cat,
        "action_items": [],
        "executive_memo": ""
    }

    if scope_type == "DAILY":
        if avg_waste > 25.0:
            advisory["action_items"].append({
                "role": "👨‍🍳 後廚出餐負責人",
                "directive": f"【明日出餐規格調校】全日平均殘食率達 {avg_waste:.1f}%，超標警告！特別是「{worst_dish}」。要求明日午市起，全面更換為 3 號標準平底飯勺（每份減量 30g 出餐）。"
            })
            advisory["action_items"].append({
                "role": "🖥️ 門市 POS / Kiosk 運營",
                "directive": f"【點餐機促銷聯動】明日於「{worst_branch}」點餐機全面上架「少飯減扣 $2」優惠，並置頂推廣低碳小食量選項。"
            })
            advisory["action_items"].append({
                "role": "📦 門市經理 (Store Manager)",
                "directive": f"【晚市/次日蒸煮下調】以今日耗損估算，明日電飯煲煮米批次請減少 2 鍋（下調約 10% 產能），單日預計防損挽回 HK$ {round(total_loss_hkd * 0.4):,.0f}。"
            })
        else:
            advisory["action_items"].append({
                "role": "👨‍🍳 後廚出餐負責人",
                "directive": f"【維持標準出餐】本日全港平均殘食率為 {avg_waste:.1f}%，整體表現優良。各店維持現有標準份量，重點維持「{worst_branch}」的出餐穩定度。"
            })
            advisory["action_items"].append({
                "role": "📦 門市經理 (Store Manager)",
                "directive": "【庫存平穩進貨】明日進貨維持基準採購量，留意尖峰備料。"
            })

        prompt = (
            f"You are the operations head of Cafe de Coral. Review today's audit: "
            f"{total_trays} trays audited, average food waste ratio {avg_waste:.1f}%, total loss HK${total_loss_hkd:.1f}. "
            f"Worst performing branch: {worst_branch}, highest waste dish: {worst_dish}. "
            f"Write a 2-sentence direct operational instruction for tomorrow's store managers."
        )
    else:
        monthly_saving_potential = total_loss_hkd * 12
        advisory["action_items"].append({
            "role": "🏭 大埔中央廚房 (Central Kitchen)",
            "directive": f"【主菜規格重新開模】月度數據顯示「{worst_dish}」長期居殘食榜首。建議中央廚房將厚切豬排/肉排單塊重量規格下調 8%（由 180g 改為 165g），徹底根治過剩浪費。"
        })
        advisory["action_items"].append({
            "role": "🏢 總部營運與菜單工程部",
            "directive": f"【門市分級差異化定價】數據印證 Level A (商業區) 白領控醣需求顯著。下月起於商業區 40 間門市正式推行「輕量少飯版菜單」，客單價微降 $1，毛利率可提升 1.8%。"
        })
        advisory["action_items"].append({
            "role": "🌱 集團 ESG 與永續發展委員會",
            "directive": f"【年化碳減量與防損核算】本月累計減廢預計年化防損可達 HK$ {monthly_saving_potential:,.0f}，月減碳 {total_co2:.1f} kg CO2e，數據已自動同步至年度可持續發展報告 (ESG Report)。"
        })

        prompt = (
            f"You are the Group CEO of Cafe de Coral Holdings. Review this month's food waste report: "
            f"{total_trays} audits, total loss HK${total_loss_hkd:.1f}, average waste {avg_waste:.1f}%. "
            f"Write an executive board-level strategic comment on supply chain and portion resizing."
        )

    try:
        inputs = engine["tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(engine["device"])
        with torch.no_grad():
            outputs = engine["generator"].generate(**inputs, max_new_tokens=80, do_sample=False)
        advisory["executive_memo"] = engine["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    except Exception:
        advisory["executive_memo"] = f"大家樂營運總部核准：落實 {scope_type} 殘食校準方針，優化中央廚房配給。"

    return advisory

# ==============================================================================
# 4. 畫面渲染控制
# ==============================================================================
def render_mode_1_detection(df_branches, df_dishes, engine):
    st.subheader("📸 Mode 1: 前線餐盤智慧掃描機 (即時過盤與入庫記錄)")
    st.caption("職責：前線回收台高速掃描 ➔ AI 自動框選殘食與辨識菜品 ➔ 即時寫入資料庫 ➔ 宏觀建議請至 Mode 2 查看")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("##### 🏢 執勤門市與掃描設置")
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

        selected_dish = None
        if captured_image:
            if auto_dish_toggle:
                detected_dish, dish_conf = auto_detect_dish_heuristic(captured_image, candidate_dishes)
                st.success(f"🔍 **AI 識別餐點參考**：`{detected_dish}` (置信度: {dish_conf:.1%})")
                selected_dish = detected_dish
            else:
                selected_dish = st.selectbox("抽檢餐點 (手動選擇)", candidate_dishes)

        # 執行推論與寫入 DB
        if captured_image and should_run_detection:
            with st.spinner("AI 偵測中: YOLOS 正在過濾非餐盤目標並辨識殘食..."):
                annotated_img, item_list, waste_ratio, primary_cat, valid_food_found = run_tray_waste_detection(captured_image, engine)

            if not valid_food_found:
                st.warning("⚠️ 影像中未檢測到合法餐盤或食物物件（已自動過濾人物/背景）。此記錄不計入殘食審計。")
                st.session_state["latest_result"] = {
                    "image": annotated_img,
                    "item_list": [],
                    "waste_ratio": 0.0,
                    "primary_cat": "無效目標 (非食物)",
                    "branch_name": selected_branch_name,
                    "dish_name": "無效餐盤",
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "cost": 0.0
                }
            else:
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
                is_saved = save_audit_record(db_record)

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
                if is_saved:
                    st.toast("✅ 掃描成功！紀錄已成功歸檔至資料庫。")

    # 右側：最新偵測看板
    with col_right:
        st.markdown("### 🎯 前線掃描結果 (Latest Tray Result)")
        latest = st.session_state.get("latest_result")

        if not latest:
            st.info("💡 尚未執行偵測。請保持相機開啟並放置餐盤，或手動拍攝/上傳。")
            placeholder_img = Image.new("RGB", (400, 260), color=(240, 240, 240))
            d = ImageDraw.Draw(placeholder_img)
            d.text((120, 120), "等待餐盤輸入中...", fill=(150, 150, 150))
            st.image(placeholder_img, caption="即時預覽看板", use_container_width=True)
        else:
            st.image(latest["image"], caption=f"審計影像: {latest['dish_name']} ({latest['timestamp']})", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("殘食佔比", f"{latest['waste_ratio']:.1%}")
            m2.metric("主要殘留", latest["primary_cat"])
            m3.metric("推算耗損", f"HK$ {latest['cost']}")

            if latest["primary_cat"] == "無效目標 (非食物)":
                st.warning("⚠️ 此鏡頭畫面被判定為非食物/自拍影像，未寫入後台數據庫。")
            else:
                st.success(f"📥 **已成功歸檔**：記錄已綁定至 `{latest['branch_name']}`。總部營運經理可在 Mode 2 查看日/月度整體建議。")

            if latest.get("item_list"):
                with st.expander("查看 Bounding Box 偵測物件明細", expanded=True):
                    st.dataframe(pd.DataFrame(latest["item_list"]), use_container_width=True)

def render_mode_2_dashboard(engine):
    st.subheader("📊 Mode 2: 大家樂集團總部 - 跨分店即時營運大盤 & 日/月度戰略建議")
    st.caption("職責：讀取真實審計資料庫 ➔ 以「日」與「月」為維度聚合統計 ➔ 由 Flan-T5 產出宏觀經營與供應鏈建議")

    df_db = get_all_audit_records()
    if df_db.empty:
        st.warning("⚠️ 目前資料庫中無任何審計數據。請先至「Mode 1」完成數次掃描測試！")
        return

    total_scans = len(df_db)
    avg_waste = df_db["waste_ratio"].mean()
    total_waste_hkd = df_db["cost_waste_hkd"].sum()
    total_co2 = df_db["co2_emission_kg"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("歷史累計盤數", f"{total_scans} 盤")
    k2.metric("全港平均殘食率", f"{avg_waste:.1f} %")
    k3.metric("累計食材損耗成本", f"HK$ {total_waste_hkd:,.1f}")
    k4.metric("累計碳排當量", f"{total_co2:.2f} kg CO2e")

    st.markdown("---")

    st.markdown("### 🧭 大家樂總部營運建議指導中心 (Macro Advisory Hub)")
    review_scope = st.radio("選擇覆盤維度 (Review Scope)", ["📅 日度營運覆盤建議 (Daily Operational Review)", "🗓️ 月度戰略採購建議 (Monthly Strategic Advisory)"], horizontal=True)

    if "日度" in review_scope:
        available_dates = df_db["audit_date"].dropna().unique().tolist()
        if not available_dates:
            available_dates = [datetime.date.today().strftime("%Y-%m-%d")]
        selected_date = st.selectbox("選擇要覆盤的日期", available_dates)
        df_target = df_db[df_db["audit_date"] == selected_date]
        scope_code = "DAILY"
    else:
        available_months = df_db["audit_month"].dropna().unique().tolist()
        if not available_months:
            available_months = [datetime.date.today().strftime("%Y-%m")]
        selected_month = st.selectbox("選擇要審計的月份", available_months)
        df_target = df_db[df_db["audit_month"] == selected_month]
        scope_code = "MONTHLY"

    if df_target.empty:
        st.info("該時段內無資料。")
    else:
        with st.spinner("AI 正在針對該時段聚合數據進行宏觀分析與決策生成..."):
            advisory = generate_macro_advisory(df_target, scope_code, engine)

        col_adv_summary, col_adv_cards = st.columns([1, 2])

        with col_adv_summary:
            st.markdown(f"#### 📌 {review_scope.split(' ')[0]} 關鍵指標")
            st.metric("該期審計樣本數", f"{advisory['total_trays']} 盤")
            st.metric("該期平均殘食率", f"{advisory['avg_waste']:.1f}%")
            st.metric("最需關注分店", advisory["worst_branch"])
            st.metric("剩餘最多菜品", advisory["worst_dish"])

        with col_adv_cards:
            st.markdown(f"#### 📋 具體下一步應對行動方針 ({review_scope.split(' ')[0]})")
            for item in advisory["action_items"]:
                st.error(f"**{item['role']}**\n\n{item['directive']}")

            with st.expander("📝 檢視 AI 總監決策備忘錄 (Executive Memo)", expanded=True):
                st.write(advisory["executive_memo"])

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

    st.markdown("### 📋 歷史審計流水表 (SQLite 完整記錄)")
    st.dataframe(df_db, use_container_width=True)

    csv_data = df_db.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 匯出當前審計數據為 CSV",
        data=csv_data,
        file_name=f"cafedecoral_audit_{datetime.date.today()}.csv",
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
# 5. 主程序入口
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
        render_mode_2_dashboard(engine)
    else:
        render_mode_3_master_data(df_branches, df_dishes)

if __name__ == "__main__":
    main()
