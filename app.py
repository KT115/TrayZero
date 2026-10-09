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
                    str(r.get("dish_name", "一哥焗豬
