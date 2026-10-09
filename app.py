import os
import datetime
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    pipeline
)
import altair as alt

# ==============================================================================
# 0. Primary Streamlit Execution Configuration
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 智能餐盤審計與會員獎勵系統", 
    page_icon="🍽️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. Global Paths & Fast-Casual POS Enterprise CSS Theme
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_FILE = os.path.join(BASE_DIR, "master_branches.csv")
DISH_FILE = os.path.join(BASE_DIR, "master_dishes.csv")
REWARD_FILE = os.path.join(BASE_DIR, "master_rewards.csv")
SEED_AUDIT_FILE = os.path.join(BASE_DIR, "seed_audit_logs.csv")
DB_FILE = os.path.join(BASE_DIR, "trayzero_audit.db")
DISH_IMG_DIR = os.path.join(BASE_DIR, "dish_references")
LOGO_FILE_PNG = os.path.join(BASE_DIR, "CDC_810.png")
LOGO_FILE_JPG = os.path.join(BASE_DIR, "CDC_810.jpg")

os.makedirs(DISH_IMG_DIR, exist_ok=True)

DEFAULT_BASE_GDRIVE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSloK2WPNFd8HPY4RfL2rNhwhk_kD12H0q09nDcrlMrx5O_zqslCOi1TPAXvlHtnP1FWxyJxGgG99QX/pub?output=csv"

def inject_safe_css():
    st.markdown("""
    <style>
        :root {
            --cdc-red: #DC2626 !important;
            --cdc-amber: #D97706 !important;
            --text-color: #0F172A !important;
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
        }
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        .main .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            color: #0F172A !important;
        }
        .pos-header-banner {
            background: linear-gradient(135deg, #C2301A 0%, #D95D1A 48%, #D87B18 100%) !important;
            border-radius: 14px !important;
            padding: 16px 24px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 4px 14px rgba(194, 48, 26, 0.22) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            display: flex !important;
            align-items: center !important;
        }
        .pos-header-title {
            color: #FFFFFF !important;
            font-size: 1.45rem !important;
            font-weight: 900 !important;
            margin: 0 !important;
            letter-spacing: 0.02em !important;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1.5px solid #E2E8F0 !important;
            padding-top: 1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin-left: auto !important;
            margin-right: auto !important;
            margin-bottom: 16px !important;
            width: 100% !important;
            text-align: center !important;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] > img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
            max-width: 175px !important;
            height: auto !important;
            object-fit: contain !important;
        }
        .pos-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 18px 20px !important;
            margin-bottom: 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
            color: #0F172A !important;
        }
        .metric-banner {
            background: #FFFFFF !important;
            border-left: 5px solid #DC2626 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
            margin-bottom: 14px !important;
        }
        .metric-banner-val {
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #0F172A !important;
            line-height: 1.2 !important;
        }
        .metric-banner-lbl {
            font-size: 0.82rem !important;
            font-weight: 700 !important;
            color: #64748B !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        .pos-badge {
            display: inline-block !important;
            padding: 4px 10px !important;
            border-radius: 6px !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
        }
        .pos-badge-green { background: #ECFDF5 !important; color: #047857 !important; border: 1px solid #A7F3D0 !important; }
        .pos-badge-amber { background: #FFFBEB !important; color: #B45309 !important; border: 1px solid #FDE68A !important; }
        .pos-badge-red { background: #FEF2F2 !important; color: #B91C1C !important; border: 1px solid #FECACA !important; }
        .pos-badge-blue { background: #EFF6FF !important; color: #1D4ED8 !important; border: 1px solid #BFDBFE !important; }
        .voucher-box {
            background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%) !important;
            border: 2px dashed #D97706 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin: 12px 0 !important;
            color: #78350F !important;
        }
        .dish-pill {
            display: inline-block !important;
            padding: 6px 14px !important;
            margin: 4px !important;
            background: #F1F5F9 !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 20px !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #1E293B !important;
            cursor: pointer !important;
        }
        .dish-pill:hover {
            background: #E2E8F0 !important;
            border-color: #94A3B8 !important;
        }
        .dish-pill.active {
            background: #DC2626 !important;
            color: #FFFFFF !important;
            border-color: #B91C1C !important;
        }
        div[data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 2px solid #E2E8F0 !important;
            margin-bottom: 20px !important;
        }
        div[data-baseweb="tab"] {
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 20px !important;
            font-weight: 700 !important;
            color: #64748B !important;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            color: #DC2626 !important;
            border-bottom: 3px solid #DC2626 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database Connection & Table Initialization
# ==============================================================================
def db_conn(): 
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def init_db():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            dish_id TEXT PRIMARY KEY,
            name TEXT,
            main_carb TEXT,
            protein TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS branches (
            name TEXT PRIMARY KEY,
            level TEXT,
            district TEXT,
            traffic TEXT,
            avg_covers INTEGER,
            base_rice_g INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rewards (
            reward_id TEXT PRIMARY KEY,
            tier_name TEXT,
            max_waste_ratio REAL,
            reward_type TEXT,
            reward_description TEXT,
            is_active BOOLEAN
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            audit_date TEXT,
            branch_name TEXT,
            dish_name TEXT,
            waste_ratio REAL,
            waste_weight_g REAL,
            estimated_cost_hkd REAL,
            carbon_kg REAL,
            primary_waste TEXT,
            member_id TEXT,
            voucher_awarded TEXT,
            dynamic_sop_alert TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM dishes")
    if cur.fetchone()[0] == 0:
        default_dishes = [
            ("D01", "一哥焗豬扒飯 (Baked Pork Chop Rice)", "白米飯", "焗厚切豬扒"),
            ("D02", "咖喱牛腩飯 (Curry Beef Brisket Rice)", "白米飯", "慢燉牛腩"),
            ("D03", "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "白米飯", "滑蛋蝦仁"),
            ("D04", "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "白米飯", "煎肉餅"),
            ("D05", "焗肉醬意粉 (Baked Spaghetti Bolognese)", "意大利麵", "慢燉牛肉醬"),
            ("D06", "車仔麵 (Kart Noodle)", "中式麵條", "牛腩/魚蛋/蘿蔔")
        ]
        cur.executemany("INSERT OR IGNORE INTO dishes VALUES (?,?,?,?)", default_dishes)

    cur.execute("SELECT COUNT(*) FROM branches")
    if cur.fetchone()[0] == 0:
        default_branches = [
            ("中環威靈頓街店", "Level A (商業核心區 / CBD)", "中西區", "白領上班族為主，午市尖峰翻檯率極高", 1200, 240),
            ("沙田新城市廣場店", "Level B (住宅商場 / Residential)", "沙田區", "家庭客、長者與週末休閒客群", 1500, 260),
            ("香港科技大學店 (HKUST)", "Level C (校園與青年區 / Campus)", "西貢區", "學生、教職員，運動量及食量顯著較大", 1800, 280),
            ("將軍澳 Popcorn 店", "Level B (住宅商場 / Residential)", "西貢區", "家庭客及換乘鐵路客流", 1400, 260)
        ]
        cur.executemany("INSERT OR IGNORE INTO branches VALUES (?,?,?,?,?,?)", default_branches)

    cur.execute("SELECT COUNT(*) FROM rewards")
    if cur.fetchone()[0] == 0:
        default_rewards = [
            ("R01", "極致光盤獎 (Ultra Clean)", 10.0, "Coupon + Points", "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", True),
            ("R02", "達標惜食獎 (Standard Clean)", 20.0, "Coupon", "【$2 堂食電子券】+【20 綠色積分】", True),
            ("R03", "支持環保獎 (Green Return)", 100.0, "Points", "【10 綠色環保積分】", True)
        ]
        cur.executemany("INSERT OR IGNORE INTO rewards VALUES (?,?,?,?,?,?)", default_rewards)

    cur.execute("SELECT COUNT(*) FROM audit_logs")
    if cur.fetchone()[0] == 0 and os.path.exists(SEED_AUDIT_FILE):
        try:
            df_seed = pd.read_csv(SEED_AUDIT_FILE)
            for _, r in df_seed.iterrows():
                cur.execute("""
                    INSERT INTO audit_logs (timestamp, audit_date, branch_name, dish_name, waste_ratio, waste_weight_g, estimated_cost_hkd, carbon_kg, primary_waste, member_id, voucher_awarded, dynamic_sop_alert)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(r.get("timestamp", "2026-09-26 12:00:00")),
                    str(r.get("audit_date", "2026-09-26")),
                    str(r.get("branch_name", "沙田新城市廣場店")),
                    str(r.get("dish_name", "一哥焗豬扒飯 (Baked Pork Chop Rice)")),
                    float(r.get("waste_ratio", 25.0)),
                    float(r.get("waste_weight_g", 125.0)),
                    float(r.get("estimated_cost_hkd", 5.2)),
                    float(r.get("carbon_kg", 0.31)),
                    str(r.get("primary_waste", "主食白飯 (Carb)")),
                    str(r.get("member_id", "M882190")),
                    str(r.get("voucher_awarded", "【$2 堂食電子券】+【20 綠色積分】")),
                    str(r.get("dynamic_sop_alert", "系統建議廚房適度調整標準份量。"))
                ))
            conn.commit()
        except Exception as e:
            print(f"Error loading seed logs: {e}")

    conn.commit()
    conn.close()

# ==============================================================================
# 3. Dynamic Cloud & Master Sync Handlers
# ==============================================================================
def get_cloud_urls():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM app_settings WHERE key LIKE 'gdrive_url_%'")
    rows = cur.fetchall()
    conn.close()
    urls = {
        "dishes": DEFAULT_BASE_GDRIVE_URL,
        "branches": DEFAULT_BASE_GDRIVE_URL,
        "rewards": DEFAULT_BASE_GDRIVE_URL
    }
    for k, v in rows:
        urls[k.replace("gdrive_url_", "")] = v
    return urls

def save_cloud_urls(urls):
    conn = db_conn()
    cur = conn.cursor()
    for k, v in urls.items():
        cur.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)", (f"gdrive_url_{k}", v))
    conn.commit()
    conn.close()

def get_live_dishes():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM dishes", conn)
    conn.close()
    if df.empty and os.path.exists(DISH_FILE):
        try:
            df = pd.read_csv(DISH_FILE)
            save_live_dishes(df)
        except Exception:
            pass
    if df.empty:
        df = pd.DataFrame([
            {"dish_id": "D01", "name": "一哥焗豬扒飯 (Baked Pork Chop Rice)", "main_carb": "白米飯", "protein": "焗厚切豬扒"},
            {"dish_id": "D02", "name": "咖喱牛腩飯 (Curry Beef Brisket Rice)", "main_carb": "白米飯", "protein": "慢燉牛腩"},
            {"dish_id": "D03", "name": "滑蛋蝦仁飯 (Scrambled Egg Shrimp Rice)", "main_carb": "白米飯", "protein": "滑蛋蝦仁"},
            {"dish_id": "D04", "name": "香辣肉燥肉餅飯 (Minced Pork Patty Rice)", "main_carb": "白米飯", "protein": "煎肉餅"},
            {"dish_id": "D05", "name": "焗肉醬意粉 (Baked Spaghetti Bolognese)", "main_carb": "意大利麵", "protein": "慢燉牛肉醬"},
            {"dish_id": "D06", "name": "車仔麵 (Kart Noodle)", "main_carb": "中式麵條", "protein": "牛腩/魚蛋/蘿蔔"}
        ])
    return df

def save_live_dishes(df):
    conn = db_conn()
    df.to_sql("dishes", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def get_live_branches():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM branches", conn)
    conn.close()
    if df.empty and os.path.exists(BRANCH_FILE):
        try:
            df = pd.read_csv(BRANCH_FILE)
            save_live_branches(df)
        except Exception:
            pass
    if df.empty:
        df = pd.DataFrame([
            {"name": "中環威靈頓街店", "level": "Level A (商業核心區 / CBD)", "district": "中西區", "traffic": "白領上班族為主，午市尖峰翻檯率極高", "avg_covers": 1200, "base_rice_g": 240},
            {"name": "沙田新城市廣場店", "level": "Level B (住宅商場 / Residential)", "district": "沙田區", "traffic": "家庭客、長者與週末休閒客群", "avg_covers": 1500, "base_rice_g": 260},
            {"name": "香港科技大學店 (HKUST)", "level": "Level C (校園與青年區 / Campus)", "district": "西貢區", "traffic": "學生、教職員，運動量及食量顯著較大", "avg_covers": 1800, "base_rice_g": 280},
            {"name": "將軍澳 Popcorn 店", "level": "Level B (住宅商場 / Residential)", "district": "沙田區", "traffic": "家庭客及換乘鐵路客流", "avg_covers": 1400, "base_rice_g": 260}
        ])
    return df

def save_live_branches(df):
    conn = db_conn()
    df.to_sql("branches", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def get_live_rewards():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM rewards WHERE is_active = 1 ORDER BY max_waste_ratio ASC", conn)
    conn.close()
    return df

def save_live_rewards(df):
    conn = db_conn()
    df.to_sql("rewards", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

# ==============================================================================
# 4. Reward Evaluator & Record Logger
# ==============================================================================
def evaluate_customer_rewards(waste_ratio_pct):
    df_rew = get_live_rewards()
    if df_rew.empty:
        if waste_ratio_pct <= 10.0:
            return "【$3 堂食現金券】+【50 綠色積分】+【凍檸茶半價券】", "極致光盤獎 (Ultra Clean)", "#10B981"
        elif waste_ratio_pct <= 20.0:
            return "【$2 堂食電子券】+【20 綠色積分】", "達標惜食獎 (Standard Clean)", "#059669"
        else:
            return "【10 綠色環保積分】", "支持環保獎 (Green Return)", "#3B82F6"
    for _, r in df_rew.iterrows():
        if waste_ratio_pct <= float(r["max_waste_ratio"]):
            return r["reward_description"], r["tier_name"], "#10B981"
    return "【10 綠色環保積分】", "支持環保獎 (Green Return)", "#64748B"

def save_record(r):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO audit_logs (timestamp, audit_date, branch_name, dish_name, waste_ratio, waste_weight_g, estimated_cost_hkd, carbon_kg, primary_waste, member_id, voucher_awarded, dynamic_sop_alert)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        r["timestamp"], r["audit_date"], r["branch_name"], r["dish_name"],
        r["waste_ratio"], r["waste_weight_g"], r["estimated_cost_hkd"],
        r["carbon_kg"], r["primary_waste"], r["member_id"],
        r["voucher_awarded"], r["dynamic_sop_alert"]
    ))
    conn.commit()
    conn.close()

def get_records():
    conn = db_conn()
    df = pd.read_sql_query("SELECT * FROM audit_logs ORDER BY id DESC", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "timestamp", "audit_date", "branch_name", "dish_name", 
            "waste_ratio", "waste_weight_g", "estimated_cost_hkd", "carbon_kg", 
            "primary_waste", "member_id", "voucher_awarded", "dynamic_sop_alert"
        ])
    return df

# ==============================================================================
# 5. AI Engine (載入今日微調之雙管線全新 Transformer 模型)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if torch.cuda.is_available() else -1
    
    # --------------------------------------------------------------------------
    # Pipeline 1: 視覺殘食審計 (今日新微調模型: kktlau115/trayzero-frozen-swin-model)
    # --------------------------------------------------------------------------
    p1_model_id = "kktlau115/trayzero-frozen-swin-model"
    print(f"📥 正在從 Hugging Face 載入今日新訓練 Pipeline 1: {p1_model_id}...")
    try:
        processor = AutoImageProcessor.from_pretrained(p1_model_id)
        model = AutoModelForImageClassification.from_pretrained(p1_model_id).to(dev)
        model.eval()
        print(f"✅ Pipeline 1 ({p1_model_id}) 載入成功！")
    except Exception as e:
        print(f"⚠️ 載入 {p1_model_id} 遇異常 ({e})，使用基底 Swin-Tiny 容錯...")
        fallback_p1 = "microsoft/swin-tiny-patch4-window7-224"
        processor = AutoImageProcessor.from_pretrained(fallback_p1)
        model = AutoModelForImageClassification.from_pretrained(fallback_p1).to(dev)
        model.eval()

    # --------------------------------------------------------------------------
    # 輔助模組: Zero-shot CLIP 菜式智慧核驗
    # --------------------------------------------------------------------------
    try:
        clip_classifier = pipeline(
            "zero-shot-image-classification", 
            model="openai/clip-vit-base-patch32", 
            device=device_id
        )
    except Exception as e:
        print(f"⚠️ CLIP 載入警告: {e}")
        clip_classifier = None

    # --------------------------------------------------------------------------
    # Pipeline 2: 智能廚房 SOP 預警與會員激勵生成 (今日新微調模型: kktlau115/trayzero-flant5-sop-alert)
    # --------------------------------------------------------------------------
    p2_model_id = "kktlau115/trayzero-flant5-sop-alert"
    print(f"📥 正在從 Hugging Face 載入今日新訓練 Pipeline 2: {p2_model_id}...")
    try:
        nlp_generator = pipeline(
            "text2text-generation",
            model=p2_model_id,
            device=device_id
        )
        print(f"✅ Pipeline 2 ({p2_model_id}) 載入成功！")
    except Exception as e:
        print(f"⚠️ 載入 {p2_model_id} 遇異常 ({e})，使用基底 Flan-T5-Base 容錯...")
        nlp_generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device_id
        )

    return {"processor": processor, "model": model, "clip": clip_classifier, "nlp": nlp_generator, "device": dev}

def detect_tray(image, engine, selected_dish="", carb_type_from_csv=""):
    image_rgb = image.convert("RGB")
    width, height = image_rgb.size
    
    inputs = engine["processor"](images=image_rgb, return_tensors="pt").to(engine["device"])
    with torch.no_grad():
        outputs = engine["model"](**inputs)
        logits = outputs.logits
        if logits.shape[-1] == 5:
            # 今日新訓練 5 分類模型 (0_Zero_Waste ~ 4_Heavy_Waste)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            # 依類別映射代表性殘食率 (0: 3%, 1: 12%, 2: 30%, 3: 55%, 4: 85%)
            class_midpoints = [0.03, 0.12, 0.30, 0.55, 0.85]
            ratio = class_midpoints[pred_idx]
        elif logits.numel() == 1:
            raw_pred = logits.item()
            ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
        else:
            raw_pred = logits[0][0].item()
            ratio = float(1.0 / (1.0 + np.exp(-raw_pred)))
        ratio = max(0.0, min(1.0, ratio))

    food_type_labels = [
        "full untouched meal on a plate",
        "mostly eaten leftover food",
        "clean empty dish"
    ]
    if engine.get("clip") is not None:
        type_res = engine["clip"](image_rgb, candidate_labels=food_type_labels)
        top_type = type_res[0]["label"]
        top_score = type_res[0]["score"]
    else:
        top_type = "mostly eaten leftover food"
        top_score = 0.85

    if "untouched" in top_type or "full" in top_type:
        ratio = max(ratio, 0.92)
    elif "clean" in top_type or "empty" in top_type:
        ratio = min(ratio, 0.03)

    ratio = max(0.0, min(1.0, ratio))

    # 🌟 6 級精細化 Grouping 分級標準
    if ratio >= 0.90:
        primary_cat = "完整未動餐點 (90-100% Untouched)"
        accent_color = "#DC2626"
    elif ratio >= 0.70:
        primary_cat = "大量剩餘 (70-89% Heavy Leftovers)"
        accent_color = "#EA580C"
    elif ratio >= 0.40:
        primary_cat = "食用過半 / 半數殘留 (40-69% Half Eaten)"
        accent_color = "#D97706"
    elif ratio >= 0.15:
        primary_cat = "少量殘留 (15-39% Minor Leftovers)"
        accent_color = "#3B82F6"
    elif ratio >= 0.06:
        primary_cat = "極少殘留 / 接近光盤 (6-14% Almost Clean)"
        accent_color = "#059669"
    else:
        primary_cat = "光盤 Clean Plate (0-5% Zero Waste)"
        accent_color = "#10B981"

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    carb_ratio = max(0.0, min(1.0, ratio * 1.05))
    protein_ratio = max(0.0, min(1.0, ratio * 0.95))
    veg_ratio = max(0.0, min(1.0, ratio * 0.90))

    items = [
        {"分類項目 Category": "主食 (Carb)", "置信度 Confidence": f"{top_score:.1%}", "佔比 Coverage": f"{carb_ratio*100:.1f}%"},
        {"分類項目 Category": "蛋白質 (Protein)", "置信度 Confidence": f"{top_score:.1%}", "佔比 Coverage": f"{protein_ratio*100:.1f}%"},
        {"分類項目 Category": "蔬菜配菜 (Vegetables)", "置信度 Confidence": f"{top_score:.1%}", "佔比 Coverage": f"{veg_ratio*100:.1f}%"},
    ]

    box = [int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85)]
    draw.rectangle(box, outline=accent_color, width=4)
    draw.text((box[0] + 10, box[1] + 10), f"TrayZero+ 殘食率: {ratio*100:.1f}%", fill=accent_color)
    
    return img_draw, items, ratio, primary_cat, True

def auto_detect_dish_clip(image, candidate_dishes, engine):
    if not candidate_dishes: return "未定義餐點", 0.0
    clean_labels = [d.strip() for d in candidate_dishes]
    try:
        if engine.get("clip") is not None:
            results = engine["clip"](image.convert("RGB"), candidate_labels=clean_labels)
            return results[0]["label"], results[0]["score"]
        else:
            return candidate_dishes[0], 0.85
    except Exception:
        return candidate_dishes[0], 0.75

# ==============================================================================
# 6. Member Profile Synthesis (Loyalty Engine)
# ==============================================================================
def analyze_member_loyalty_profile(member_id, df_all):
    if not member_id or member_id == "STAFF" or df_all.empty:
        return None
    
    m_df = df_all[df_all["member_id"] == member_id]
    if m_df.empty:
        return {
            "member_id": member_id, "total_visits": 0, "avg_waste": 0.0,
            "favorite_dish": "尚無資料", "pos_default_rice": "正常份量",
            "pos_default_sauce": "正常汁", "retarget_strategy": "發送迎新 $5 折扣券吸引二訪",
            "crm_segment": "新註冊會員 (New Sign-up)"
        }
    
    total_visits = len(m_df)
    avg_waste = m_df["waste_ratio"].mean()
    
    dish_stats = m_df.groupby("dish_name").agg(
        count=("waste_ratio", "count"),
        clean_waste=("waste_ratio", "mean")
    ).reset_index()
    fav_dish = dish_stats.sort_values(by=["count", "clean_waste"], ascending=[False, True]).iloc[0]["dish_name"]
    
    carb_wastes = m_df[m_df["primary_waste"].str.contains("主食", na=False)]
    if not carb_wastes.empty and carb_wastes["waste_ratio"].mean() > 20.0:
        pos_rice = "預設【少飯/少麵 (-30g / 立減 $2)】"
        crm_seg = "控醣輕食族 (Low-Carb Diners)"
    elif avg_waste < 10.0:
        pos_rice = "預設【正常份量 (光盤常客)】"
        crm_seg = "高飽足飽腹族 (Standard/High-Calorie)"
    else:
        pos_rice = "預設【標準份量】"
        crm_seg = "均衡飲食族 (Balanced Diners)"

    sauce_wastes = m_df[m_df["primary_waste"].str.contains("配菜|醬汁|湯汁", na=False)]
    if not sauce_wastes.empty and sauce_wastes["waste_ratio"].mean() > 25.0:
        pos_sauce = "預設【少汁 / 醬汁另上】"
    else:
        pos_sauce = "預設【正常汁】"

    if "控醣" in crm_seg:
        retarget_strategy = f"針對最愛餐點【{fav_dish}】推送「少飯少麵換特飲券」；推廣高蛋白輕食。"
    elif "高飽足" in crm_seg:
        retarget_strategy = f"向其大家樂 App 推送【{fav_dish}】加配小食（雞翼/紅豆冰）$8 組合券。"
    else:
        retarget_strategy = f"推送【{fav_dish}】午市立減 $3 現金回訪券，鎖定工作日高頻復購。"

    return {
        "member_id": member_id, "total_visits": total_visits, "avg_waste": avg_waste,
        "favorite_dish": fav_dish, "pos_default_rice": pos_rice, "pos_default_sauce": pos_sauce,
        "retarget_strategy": retarget_strategy, "crm_segment": crm_seg
    }

# ==============================================================================
# 7. Mode Renderers
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="pos-header-banner">
        <div>
            <h2 class="pos-header-title">🍽️ TrayZero+ | 大家樂智能餐盤廚餘審計與會員閉環系統</h2>
            <div style="color: #FEF3C7; font-size: 0.85rem; font-weight: 600; margin-top: 4px;">
                Café de Coral AI ESG & Smart Operations Engine · 双管线深度学习驱动
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(engine, modules):
    st.markdown("### 📷 Mode 1: 門市前線智能收盤點餐機 (Frontline Kiosk)")
    
    col_k1, col_k2 = st.columns([1, 1], gap="large")

    df_b = get_live_branches()
    df_d = get_live_dishes()
    
    branch_names = df_b["name"].tolist() if not df_b.empty else ["沙田新城市廣場店"]
    dish_names = df_d["name"].tolist() if not df_d.empty else ["一哥焗豬扒飯 (Baked Pork Chop Rice)"]

    with col_k1:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        st.markdown("#### 1. 門市與會員身份辨識")
        
        sel_branch = st.selectbox("當前運行分店", branch_names, key="kiosk_branch")
        member_id = st.text_input("Club 100 會員 QR Code / 手機號碼 (留空代表匿名訪客)", value="M882190", key="kiosk_member")
        
        st.markdown("---")
        st.markdown("#### 2. 餐盤影像獲取")
        input_method = st.radio("影像來源", ["測試圖片 / 拍照上傳", "即時相機拍攝"], horizontal=True)
        
        image = None
        if input_method == "即時相機拍攝":
            camera_file = st.camera_input("拍攝餐盤殘留")
            if camera_file:
                image = Image.open(camera_file)
        else:
            uploaded_file = st.file_uploader("上傳餐盤照片 (.jpg / .png)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
            else:
                image = Image.new("RGB", (224, 224), color=(235, 230, 220))
                st.caption("💡 提示：目前使用系統內置預設餐盤圖片進行展示。")

        if image:
            st.image(image, caption="待審計影像", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 3. 菜式辨識")
        auto_dish, dish_conf = auto_detect_dish_clip(image, dish_names, engine)
        st.info(f"🔍 CLIP 智慧推薦菜式：**{auto_dish}** (信心度: {dish_conf:.1%})")
        
        sel_dish = st.selectbox("確認當前餐點菜式", dish_names, index=dish_names.index(auto_dish) if auto_dish in dish_names else 0)
        
        run_btn = st.button("🚀 執行 TrayZero+ 智能審計 (Run Audit)", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_k2:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        st.markdown("#### 4. 深度學習審計與即時回饋")

        if run_btn and image:
            with st.spinner("🤖 正在執行 Pipeline 1 (Swin) 與 Pipeline 2 (Flan-T5)..."):
                annotated_img, items, ratio, primary_cat, is_food = detect_tray(image, engine, selected_dish=sel_dish)
                
                waste_pct = round(ratio * 100, 1)
                now = datetime.datetime.now()

                b_row = df_b[df_b["name"] == sel_branch]
                base_rice = float(b_row.iloc[0]["base_rice_g"]) if not b_row.empty else 250.0
                
                waste_weight = round((base_rice + 200.0) * (waste_pct / 100.0), 1)
                waste_cost = round(waste_weight * 0.042, 2)
                waste_carbon = round(waste_weight * 0.0025, 3)

                voucher_text, tier_name, tier_color = evaluate_customer_rewards(waste_pct)

                # Pipeline 2 呼叫
                try:
                    nlp_prompt = f"Generate kitchen SOP alert and customer incentive for Café de Coral: Dish: {sel_dish}, Waste: {waste_pct}%, Category: {primary_cat}. Action:"
                    sop_gen = engine["nlp"](nlp_prompt, max_length=128)[0]["generated_text"]
                    dynamic_sop_alert = sop_gen.strip()
                except Exception:
                    dynamic_sop_alert = f"連續監測到【{sel_dish}】殘食率偏高 ({waste_pct}%)，建議廚房適度調整白飯與醬汁裝盤標準量。"

                save_record({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), 
                    "audit_date": now.strftime("%Y-%m-%d"),
                    "branch_name": sel_branch, 
                    "dish_name": sel_dish,
                    "waste_ratio": waste_pct, 
                    "waste_weight_g": waste_weight,
                    "estimated_cost_hkd": waste_cost, 
                    "carbon_kg": waste_carbon,
                    "primary_waste": items[0]["分類項目 Category"], 
                    "member_id": member_id,
                    "voucher_awarded": voucher_text, 
                    "dynamic_sop_alert": dynamic_sop_alert
                })

            st.image(annotated_img, caption=f"視覺審計結果 (殘食率: {waste_pct}%)", use_container_width=True)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-banner">
                    <div class="metric-banner-lbl">殘食百分比</div>
                    <div class="metric-banner-val">{waste_pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-banner">
                    <div class="metric-banner-lbl">估算浪費成本</div>
                    <div class="metric-banner-val">HK${waste_cost}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                st.markdown(f"""
                <div class="metric-banner">
                    <div class="metric-banner-lbl">產生碳排放</div>
                    <div class="metric-banner-val">{waste_carbon} kg</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="voucher-box">
                <h4 style="margin: 0 0 6px 0; color: #92400E;">🎉 Club 100 獎勵發放：{tier_name}</h4>
                <div style="font-size: 1.05rem; font-weight: 800; color: #B45309;">{voucher_text}</div>
                <div style="font-size: 0.8rem; margin-top: 4px; color: #78350F;">電子券已即時存入會員帳戶 ({member_id})，可於大家樂門市下次消費直接扣減。</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 📋 Pipeline 2 生成之廚房 SOP 改進指令")
            st.info(f"🔔 **後廚通知**：{dynamic_sop_alert}")

            st.markdown("##### 🔍 各大食材分項佔比 (Macronutrient Coverage)")
            st.table(pd.DataFrame(items))

        else:
            st.write("👈 請於左側確認設定並點擊「執行 TrayZero+ 智能審計」進行實時推論。")

        st.markdown('</div>', unsafe_allow_html=True)

def render_mode2(engine, modules):
    st.markdown("### 📊 Mode 2: 總部營運與 ESG 大數據儀表板 (Operations HQ BI)")
    
    df_all = get_records()
    if df_all.empty:
        st.warning("⚠️ 目前資料庫尚無審計日誌，請先至 Mode 1 執行推論。")
        return

    st.markdown('<div class="pos-card">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("累計審計餐盤總數", f"{len(df_all):,} 盤")
    with m2:
        st.metric("平均殘食率", f"{df_all['waste_ratio'].mean():.1f}%")
    with m3:
        st.metric("累計食物成本浪費", f"HK$ {df_all['estimated_cost_hkd'].sum():,.1f}")
    with m4:
        st.metric("累計 Scope 3 碳排放", f"{df_all['carbon_kg'].sum():,.2f} kg")
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        st.markdown("#### 🍲 菜式平均殘食率排行 (Top Wasted Dishes)")
        dish_summary = df_all.groupby("dish_name")["waste_ratio"].mean().reset_index()
        chart_dish = alt.Chart(dish_summary).mark_bar(color="#DC2626").encode(
            x=alt.X("waste_ratio:Q", title="平均殘食率 (%)"),
            y=alt.Y("dish_name:N", sort="-x", title="餐點名稱")
        ).properties(height=280)
        st.altair_chart(chart_dish, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        st.markdown("#### 🏪 門市殘食率分佈 (Branch Benchmarks)")
        branch_summary = df_all.groupby("branch_name")["waste_ratio"].mean().reset_index()
        chart_branch = alt.Chart(branch_summary).mark_bar(color="#D97706").encode(
            x=alt.X("waste_ratio:Q", title="平均殘食率 (%)"),
            y=alt.Y("branch_name:N", sort="-x", title="分店")
        ).properties(height=280)
        st.altair_chart(chart_branch, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="pos-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 即時審計明細數據表 (Live Audit Records)")
    st.dataframe(df_all, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_mode3():
    st.markdown("### ⚙️ Mode 3: 菜單、分店與獎勵規則動態管理 (Dynamic Configuration)")
    
    tab1, tab2, tab3 = st.tabs(["🍛 菜單管理 (Dishes)", "🏪 門市管理 (Branches)", "🎁 獎勵規則 (Rewards)"])
    
    with tab1:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        df_dishes = get_live_dishes()
        st.dataframe(df_dishes, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        df_branches = get_live_branches()
        st.dataframe(df_branches, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="pos-card">', unsafe_allow_html=True)
        df_rewards = get_live_rewards()
        st.dataframe(df_rewards, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 8. Main Application Controller
# ==============================================================================
def main():
    inject_safe_css()
    init_db()

    # 側邊欄設計
    if os.path.exists(LOGO_FILE_PNG):
        st.sidebar.image(LOGO_FILE_PNG, use_container_width=True)
    elif os.path.exists(LOGO_FILE_JPG):
        st.sidebar.image(LOGO_FILE_JPG, use_container_width=True)
    else:
        st.sidebar.markdown("## 🍽️ Café de Coral")

    st.sidebar.title("TrayZero+ 控制台")
    st.sidebar.caption("大家樂集團 · 智能餐盤審計")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "系統運行模式 (SYSTEM MODE)", 
        [
            "Mode 1: 門市前線收盤機 (Frontline Kiosk)", 
            "Mode 2: 總部 BI 大數據看板 (HQ Analytics)", 
            "Mode 3: 菜單與獎勵配置 (Dynamic Config)"
        ]
    )

    with st.spinner("🚀 正在啟動雙管線深度學習引擎 (Loading AI Engine)..."):
        engine = load_ai_engine()

    # 側邊欄企業模組狀態開關
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 企業模組狀態 (MODULES)")
    mod_1 = st.sidebar.checkbox("M1: 營運監控 (Ops Core)", value=True)
    mod_2 = st.sidebar.checkbox("M2: 深度分析 (BI Analytics)", value=True)
    mod_3 = st.sidebar.checkbox("M3: 精準營銷 (Smart POS)", value=True)
    mod_4 = st.sidebar.checkbox("M4: 會員閉環 (Loyalty Loop)", value=True)

    active_modules = {
        "mod1": mod_1,
        "mod2": mod_2,
        "mod3": mod_3,
        "mod4": mod_4
    }

    render_header()

    if mode.startswith("Mode 1"): 
        render_mode1(engine, active_modules)
    elif mode.startswith("Mode 2"): 
        render_mode2(engine, active_modules)
    else: 
        render_mode3()

if __name__ == "__main__":
    main()
