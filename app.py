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
    AutoModelForObjectDetection, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

# ==============================================================================
# 0. Global Constants & Configuration
# ==============================================================================
BRANCH_FILE = "master_branches.csv"
DISH_FILE = "master_dishes.csv"
DB_FILE = "trayzero_audit.db"
LOGO_FILE = "CDC_810.png"

FOOD_WHITELIST = {
    "bowl": "Rice", "cake": "Rice", "sandwich": "Meat", "pizza": "Meat", "hot dog": "Meat",
    "carrot": "Veg_Soup", "broccoli": "Veg_Soup", "apple": "Veg_Soup", "orange": "Veg_Soup",
    "donut": "Meat", "cup": "Veg_Soup", "bottle": "Veg_Soup", "dining table": "Tray"
}

# ==============================================================================
# 1. Clean UI Theme + Frontend Flow Stepper Styles
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700;900&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Noto Sans TC', sans-serif;
        }

        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #334155 !important;
            font-weight: 500;
        }

        /* Force Sidebar Image Centering */
        [data-testid="stSidebar"] [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            margin: 0 auto !important;
            text-align: center !important;
        }

        [data-testid="stSidebar"] [data-testid="stImage"] img {
            margin: 0 auto !important;
            display: block !important;
        }

        /* Minimalist Single-Line Header */
        .trayzero-header {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 18px 26px;
            margin-bottom: 16px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.03);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .trayzero-title {
            color: #0F172A !important;
            font-size: 1.25rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.01em;
            margin: 0 !important;
            white-space: nowrap !important;
        }

        /* Frontend Flow Stepper Container */
        .flow-stepper-container {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 12px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
        }

        .flow-step-item {
            display: flex;
            align-items: center;
            font-size: 0.82rem;
            font-weight: 600;
            color: #64748B;
        }
        .flow-step-active {
            color: #2563EB;
            font-weight: 700;
        }
        .flow-step-num {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #F1F5F9;
            color: #64748B;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            margin-right: 8px;
            font-weight: 800;
        }
        .flow-step-num-active {
            background: #2563EB;
            color: #FFFFFF;
        }
        .flow-arrow {
            color: #CBD5E1;
            font-weight: bold;
            font-size: 0.85rem;
        }

        /* Clean Micro-Elevated Cards */
        .clean-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.03);
            transition: all 0.2s ease-in-out;
        }
        .clean-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.06);
            transform: translateY(-2px);
        }

        .clean-label {
            font-size: 0.74rem;
            color: #64748B;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .clean-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }

        /* Directives & Action Cards */
        .directive-card {
            border-radius: 14px;
            padding: 16px 20px;
            margin-bottom: 12px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #CBD5E1;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
        }
        .directive-chef { border-left-color: #EF4444; }
        .directive-pos { border-left-color: #F59E0B; }
        .directive-mgr { border-left-color: #3B82F6; }

        .directive-title {
            font-weight: 700;
            font-size: 0.88rem;
            color: #0F172A;
            margin-bottom: 4px;
        }

        .directive-body {
            font-size: 0.84rem;
            color: #475569;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. Database & Data Storage Layer
# ==============================================================================
def db_conn(): 
    return sqlite3.connect(DB_FILE)

def init_db():
    with db_conn() as conn:
        conn.execute("""
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
        cols = [c[1] for c in conn.execute("PRAGMA table_info(audit_logs)").fetchall()]
        if "audit_date" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_date TEXT")
        if "audit_month" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_month TEXT")

def save_record(r):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO audit_logs (
                timestamp, audit_date, audit_month, branch_name, branch_level,
                dish_name, primary_waste, waste_ratio, cost_waste_hkd, co2_emission_kg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["timestamp"], r["audit_date"], r["audit_month"], r["branch_name"], r["branch_level"],
            r["dish_name"], r["primary_waste"], r["waste_ratio"], r["cost_waste_hkd"], r["co2_emission_kg"]
        ))

def get_records():
    with db_conn() as conn: 
        return pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)

def reset_db():
    with db_conn() as conn: 
        conn.execute("DELETE FROM audit_logs")

def load_master_data():
    df_b = pd.read_csv(BRANCH_FILE) if os.path.exists(BRANCH_FILE) else pd.DataFrame(
        columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"]
    )
    df_d = pd.read_csv(DISH_FILE) if os.path.exists(DISH_FILE) else pd.DataFrame(
        columns=["dish_id", "name", "main_carb", "protein"]
    )
    return df_b, df_d

# ==============================================================================
# 3. AI Inference Engine (PyTorch & Transformers)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_ai_engine():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_path = "./Fine-tuned_Model_files" if (os.path.exists("./Fine-tuned_Model_files") and any(os.scandir("./Fine-tuned_Model_files"))) else "hustvl/yolos-tiny"
    return {
        "proc": AutoImageProcessor.from_pretrained(m_path),
        "det": AutoModelForObjectDetection.from_pretrained(m_path).to(dev),
        "tok": AutoTokenizer.from_pretrained("google/flan-t5-base"),
        "gen": AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(dev),
        "device": dev
    }

def detect_tray(image, engine):
    inp = engine["proc"](images=image, return_tensors="pt").to(engine["device"])
    with torch.no_grad(): 
        out = engine["det"](**inp)
        
    sz = torch.tensor([image.size[::-1]]).to(engine["device"])
    res = engine["proc"].post_process_object_detection(out, threshold=0.20, target_sizes=sz)[0]
    
    img_draw = image.copy()
    total_area = image.size[0] * image.size[1]
    waste_area = 0
    draw = ImageDraw.Draw(img_draw)
    items, valid_food = [], False
    color_map = {"Rice": "#EF4444", "Meat": "#F59E0B", "Veg_Soup": "#10B981"}
    primary = "光盤 Clean Plate"

    for box, score, label_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        lbl = engine["det"].config.id2label.get(label_id, "item")
        if lbl not in FOOD_WHITELIST: 
            continue
            
        cat = FOOD_WHITELIST[lbl]
        valid_food = True
        
        if cat == "Rice": 
            name, primary = "白飯殘留 Rice Waste", "白飯殘留 Rice Residual"
        elif cat == "Meat":
            name = "肉類殘留 Meat Residual"
            if "白飯" not in primary: 
                primary = "肉類殘留 Protein Residual"
        elif cat == "Veg_Soup":
            name = f"配菜/醬汁 Sides ({lbl})"
            if primary == "光盤 Clean Plate": 
                primary = "配菜/醬汁 Sides & Sauce"
        else: 
            name = "餐盤基準 Tray Baseline"

        b = [max(0, box[0]), max(0, box[1]), min(image.size[0], box[2]), min(image.size[1], box[3])]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if cat != "Tray": 
            waste_area += area

        c = color_map.get(cat, "#3B82F6")
        draw.rectangle(b, outline=c, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{name} {score:.0%}", fill=c)
        items.append({
            "分類項目 Category": name, 
            "置信度 Confidence": f"{score:.1%}", 
            "佔比 Coverage": f"{area/total_area:.1%}"
        })

    ratio = min(1.0, waste_area / (total_area * 0.65)) if (total_area > 0 and valid_food) else 0.0
    return img_draw, items, ratio, primary, valid_food

def auto_detect_dish(image, candidate_dishes):
    if not candidate_dishes: 
        return "未定義餐點 Undefined Dish", 0.0
    np_img = np.array(image.resize((32, 32)))
    r, g, b = np.mean(np_img[:, :, 0]), np.mean(np_img[:, :, 1]), np.mean(np_img[:, :, 2])
    idx = 1 if (r > 140 and g > 110 and b < 90 and len(candidate_dishes) > 1) else 0
    return candidate_dishes[idx], 0.88

# ==============================================================================
# 4. Macro Advisory Synthesis Engine (Bi-lingual)
# ==============================================================================
def get_advisory(df, scope_type, branch_sel, dish_sel, engine):
    if df.empty:
        n, avg_w, loss, co2 = 0, 0.0, 0.0, 0.0
        t_branch = branch_sel if branch_sel != "ALL" else "全港分店 All Branches"
        t_dish = dish_sel if dish_sel != "ALL" else "全部餐點 All Menu Items"
    else:
        n = len(df)
        avg_w = df["waste_ratio"].mean()
        loss = df["cost_waste_hkd"].sum()
        co2 = df["co2_emission_kg"].sum()
        t_branch = branch_sel if branch_sel != "ALL" else df.groupby("branch_name")["waste_ratio"].mean().idxmax()
        t_dish = dish_sel if dish_sel != "ALL" else df.groupby("dish_name")["waste_ratio"].mean().idxmax()

    actions = [
        {
            "type": "directive-chef", 
            "role": f"👨‍🍳 後廚出餐負責人 Head Chef ({t_branch} • {t_dish})",
            "text": f"【即時出餐規格調校 Portion Resizing】平均殘食率達 {avg_w:.1f}%。即刻針對「{t_dish}」換裝 3 號標準平底飯勺（每份減量 30g 出餐），防止熟米過剩積壓。\n"
                    f"(Average plate waste is {avg_w:.1f}%. Immediately switch to a standard size-3 portion scoop (-30g per serving) on {t_dish} to eliminate prep backlog.)"
        },
        {
            "type": "directive-pos", 
            "role": f"🖥️ 門市 POS / Kiosk 促銷營運 Front-of-House Promotion",
            "text": f"【點餐機輕量促銷聯動 Kiosk Promo】於「{t_branch}」自助點餐機置頂彈窗提示「少飯減扣 $2」優惠，引流小食量顧客主動選擇輕量裝。\n"
                    f"(Activate automated POS prompt offering 'Light Portion (-HK$2)' for {t_dish} at {t_branch} to guide low-appetite diners toward right-sized meals.)"
        },
        {
            "type": "directive-mgr", 
            "role": f"📦 門市經理與採購部 Store Manager & Sourcing",
            "text": f"【蒸煮備料與減碳核算 Supply Prep】電飯煲次輪蒸煮批次下調 10%。單期預計防損挽回 HK$ {max(150, round(loss * 0.4)):,.0f}，碳減量 {co2:.1f} kg CO2e。\n"
                    f"(Reduce cooked rice batches by 10%. Projected waste prevention: HK$ {max(150, round(loss * 0.4)):,.0f}; GHG mitigation: {co2:.1f} kg CO2e.)"
        }
    ]
    
    try:
        p = f"You are CEO of Cafe de Coral. Review: {n} audited trays, average waste {avg_w:.1f}%, estimated loss HK${loss:.0f} across {t_branch} for {t_dish}. Provide one concise board-level operational instruction."
        inp = engine["tok"](p, return_tensors="pt", max_length=256, truncation=True).to(engine["device"])
        memo = engine["tok"].decode(engine["gen"].generate(**inp, max_new_tokens=60)[0], skip_special_tokens=True)
    except Exception:
        memo = f"核准：落實 {scope_type} 殘食校準方針，精準優化各門市配給量。(Approved: Execute {scope_type} portion calibration policy.)"

    return {"total": n, "avg_w": avg_w, "branch": t_branch, "dish": t_dish, "actions": actions, "memo": memo}

# ==============================================================================
# 5. UI Views & Frontend Flow Stepper
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="trayzero-header">
        <h2 class="trayzero-title">🍽️ TrayZero 智能餐盤殘食審計與中央調配系統 (Intelligent Plate Waste Auditing System)</h2>
    </div>
    """, unsafe_allow_html=True)

def render_frontend_stepper(current_step: int = 1):
    """前端視覺化流程指示條 (Frontend Flow Stepper)"""
    s1_cls = "flow-step-active" if current_step >= 1 else ""
    s2_cls = "flow-step-active" if current_step >= 2 else ""
    s3_cls = "flow-step-active" if current_step >= 3 else ""
    s4_cls = "flow-step-active" if current_step >= 4 else ""

    n1_cls = "flow-step-num-active" if current_step >= 1 else ""
    n2_cls = "flow-step-num-active" if current_step >= 2 else ""
    n3_cls = "flow-step-num-active" if current_step >= 3 else ""
    n4_cls = "flow-step-num-active" if current_step >= 4 else ""

    st.markdown(f"""
    <div class="flow-stepper-container">
        <div class="flow-step-item {s1_cls}">
            <span class="flow-step-num {n1_cls}">1</span> 畫面輸入 (Ingestion)
        </div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step-item {s2_cls}">
            <span class="flow-step-num {n2_cls}">2</span> 門禁過濾 (Quality Gate)
        </div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step-item {s3_cls}">
            <span class="flow-step-num {n3_cls}">3</span> AI 分割分析 (Inference)
        </div>
        <div class="flow-arrow">&rarr;</div>
        <div class="flow-step-item {s4_cls}">
            <span class="flow-step-num {n4_cls}">4</span> 看板歸檔 (Sync & Report)
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(df_b, df_d, engine):
    # 呈現前端動態流程步驟條
    current_step = 1
    if st.session_state.get("latest") is not None:
        current_step = 4
    render_frontend_stepper(current_step)

    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請先切換至「Mode 3」上傳 CSV。\n(Store or menu database is empty. Please navigate to 'Mode 3' to bulk upload CSV files.)")
        return

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("#### 🏢 執勤門市與掃描設定 (Station & Input Settings)")
        b_name = st.selectbox("執勤門市 (Active Store Location)", df_b["name"].tolist())
        b_meta = df_b[df_b["name"] == b_name].iloc[0]
        st.caption(f"等級 Tier: `{b_meta['level']}` | 區域 District: `{b_meta['district']}` | 標配飯量 Standard: `{b_meta['base_rice_g']}g`")

        auto_dish = st.checkbox("🤖 啟用 AI 自動辨識餐點類型 (Auto Dish Recognition)", value=True)
        scan_mode = st.radio(
            "掃描模式 (Scanning Method)", 
            ["🟢 Live Camera 長開 (靜止自動感應 / Auto-Scan)", "📸 手動快照 (Manual Snapshot)", "📁 上傳照片 (Upload Image)"], 
            horizontal=True
        )
        
        img_cap, do_scan = None, False
        if scan_mode.startswith("🟢"):
            cam = st.camera_input("持續監控畫面 (Live Feed Monitor)", key="live_cam")
            if cam:
                img_cap = Image.open(cam).convert("RGB")
                h = hash(img_cap.tobytes()[:3000])
                if h != st.session_state.get("last_h"):
                    st.session_state["last_h"], do_scan = h, True
                else: 
                    st.info("🟢 監控中：當前餐盤已完成分析，等待更換餐盤...\n(Monitoring: Active tray already analyzed. Awaiting next tray...)")
        elif scan_mode.startswith("📸"):
            m_cam = st.camera_input("拍照 (Take Snapshot)", key="manual_cam")
            if m_cam: 
                img_cap, do_scan = Image.open(m_cam).convert("RGB"), True
        else:
            up = st.file_uploader("上傳餐盤相片 (Upload Tray Image)", type=["jpg", "png", "jpeg"], key="tray_file_uploader")
            if up is not None:
                img_cap = Image.open(up).convert("RGB")
                file_sig = f"{up.name}_{up.size}"
                if file_sig != st.session_state.get("last_uploaded_sig"):
                    st.session_state["last_uploaded_sig"] = file_sig
                    do_scan = True

        # 執行推論與流程推進
        if img_cap and do_scan:
            with st.spinner("AI 偵測中: YOLOS 正在過濾非餐盤目標並辨識殘食..."):
                anno_img, items, ratio, primary_cat, is_food = detect_tray(img_cap, engine)

            if not is_food:
                st.error("🚫 偵測失敗：未檢測到合法餐盤或食物物件！（已自動過濾人物/背景）\n(Detection Failed: No valid tray or food objects detected! People/backgrounds filtered.)")
                st.session_state["latest"] = None
            else:
                sel_dish, _ = auto_detect_dish(img_cap, df_d["name"].tolist()) if auto_dish else (st.selectbox("指定餐點 (Select Target Dish)", df_d["name"].tolist()), 1.0)
                loss_hkd = round(ratio * 25 * 0.45, 1)
                now = datetime.datetime.now()
                save_record({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), 
                    "audit_date": now.strftime("%Y-%m-%d"),
                    "audit_month": now.strftime("%Y-%m"), 
                    "branch_name": b_name, 
                    "branch_level": str(b_meta['level']).split(" ")[0],
                    "dish_name": sel_dish, 
                    "primary_waste": primary_cat, 
                    "waste_ratio": round(ratio * 100, 1),
                    "cost_waste_hkd": loss_hkd, 
                    "co2_emission_kg": round(loss_hkd * 0.12, 2)
                })
                st.session_state["latest"] = {
                    "img": anno_img, "dish": sel_dish, "time": now.strftime("%H:%M:%S"),
                    "ratio": ratio, "cat": primary_cat, "cost": loss_hkd, "branch": b_name, "items": items
                }
                st.toast("✅ 審計記錄成功歸檔！(Tray audit recorded and indexed!)")
                st.rerun()

    with c2:
        st.markdown("#### 🎯 前線掃描結果 (Latest Scan Result)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測或畫面非餐盤。請對準餐盤掃描。\n(No valid tray scan available. Align camera with collection tray.)")
        else:
            st.image(latest["img"], caption=f"{latest['dish']} ({latest['time']})", use_container_width=True)
            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="clean-card"><div class="clean-label">殘食佔比 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if latest["ratio"] > 0.3 else "#10B981"}">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="clean-card"><div class="clean-label">主要殘留 Primary Residual</div><div class="clean-val" style="font-size:1.1rem;margin-top:6px;">{latest["cat"].split(" ")[0]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="clean-card"><div class="clean-label">推算損耗 Loss (HK$)</div><div class="clean-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)
            st.success(f"📥 **記錄已歸檔 Logged**: `{latest['branch']}`")

def render_mode2(df_b, df_d, engine):
    st.markdown("### 📊 總部即時營運大盤 & 戰略建議 (Executive Overview & Strategic Advisory)")
    df_raw = get_records()

    st.markdown("#### 🎛️ 雙軸分析維度 (Analysis Dimensions)")
    c1, c2 = st.columns(2)
    with c1:
        b_filter = st.selectbox(
            "1. 門市維度過濾 (Store Location Dimension)", 
            ["🌐 全部分店 (Overall Branches)"] + df_b["name"].tolist() if not df_b.empty else ["🌐 全部分店 (Overall Branches)"]
        )
        sel_b = "ALL" if "全部" in b_filter else b_filter
    with c2:
        d_filter = st.selectbox(
            "2. 食物種類維度過濾 (Menu Item Dimension)", 
            ["🍱 全部餐點品項 (Overall Menu Items)"] + df_d["name"].tolist() if not df_d.empty else ["🍱 全部餐點品項 (Overall Menu Items)"]
        )
        sel_d = "ALL" if "全部" in d_filter else d_filter

    df_filtered = df_raw.copy()
    if not df_filtered.empty:
        if sel_b != "ALL": 
            df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL": 
            df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    # KPI Statistics
    n = len(df_filtered)
    avg_w = df_filtered["waste_ratio"].mean() if n > 0 else 0.0
    tot_hkd = df_filtered["cost_waste_hkd"].sum() if n > 0 else 0.0
    tot_co2 = df_filtered["co2_emission_kg"].sum() if n > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="clean-card"><div class="clean-label">審計樣本盤數 Audited Trays</div><div class="clean-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div><div class="clean-sub">即時同步 Real-time Sync</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="clean-card"><div class="clean-label">平均殘食率 Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if avg_w > 25 else "#10B981"}">{avg_w:.1f}%</div><div class="clean-sub">基準目標 Target: &lt;15%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="clean-card"><div class="clean-label">食材損耗總額 Total Loss</div><div class="clean-val" style="color:#F59E0B">HK${tot_hkd:,.1f}</div><div class="clean-sub">動態估算 Dynamic Valuation</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="clean-card"><div class="clean-label">累計碳排放 GHG Emissions</div><div class="clean-val" style="color:#3B82F6">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div><div class="clean-sub">Scope 3 ESG Metric</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    scope = st.radio("覆盤時限 (Advisory Scope)", ["📅 日度營運覆盤建議 (Daily Operational Review)", "🗓️ 月度戰略採購建議 (Monthly Strategic Advisory)"], horizontal=True)
    scope_code = "DAILY" if "日度" in scope else "MONTHLY"
    date_col = "audit_date" if scope_code == "DAILY" else "audit_month"
    
    dates = df_filtered[date_col].dropna().unique().tolist() if not df_filtered.empty else [datetime.date.today().strftime("%Y-%m-%d" if scope_code=="DAILY" else "%Y-%m")]
    s_date = st.selectbox(f"選擇審計{'日期' if scope_code=='DAILY' else '月份'} (Select Audit {'Date' if scope_code=='DAILY' else 'Month'})", dates)
    df_scope = df_filtered[df_filtered[date_col] == s_date] if not df_filtered.empty else pd.DataFrame()

    with st.spinner("AI 正在分析生成營運指引... (Generating executive recommendations...)"):
        adv = get_advisory(df_scope, scope_code, sel_b, sel_d, engine)

    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="clean-label">當期指標摘要 (Scope Summary)</div>
            <div style="font-size:0.88rem;color:#334155;line-height:1.8;margin-top:8px;">
                • 審計盤數 Audited Trays: <b>{adv['total']} 盤</b><br>
                • 殘食率 Waste Ratio: <b style="color:#EF4444">{adv['avg_w']:.1f}%</b><br>
                • 目標門市 Target Branch: <b>{adv['branch']}</b><br>
                • 目標餐點 Target Dish: <b>{adv['dish']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        for a in adv["actions"]:
            st.markdown(f'<div class="directive-card {a["type"]}"><div class="directive-title">{a["role"]}</div><div class="directive-body">{a["text"]}</div></div>', unsafe_allow_html=True)
        with st.expander("📝 檢視 AI 總監決策備忘錄 (View AI Executive Memo)", expanded=True):
            st.write(adv["memo"])

    st.markdown("---")
    if not df_filtered.empty:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🏢 各門市平均殘食率 Store Waste Ratio (%)")
            st.bar_chart(df_filtered.groupby("branch_name")["waste_ratio"].mean(), color="#3B82F6")
        with g2:
            st.markdown("##### 🍱 各食物種類耗損 Waste Cost by Dish (HK$)")
            st.bar_chart(df_filtered.groupby("dish_name")["cost_waste_hkd"].sum(), color="#EF4444")
        st.markdown("##### 📋 當前維度流水表 (Active Audit Records)")
        st.dataframe(df_filtered, use_container_width=True)
        
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 匯出當前維度數據 (Export Active CSV)",
            data=csv_data,
            file_name=f"trayzero_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 目前所選維度尚無過盤紀錄。(No audit records found for the selected dimensions.)")

def render_mode3(df_b, df_d):
    st.markdown("### ⚙️ 基礎資料管理 (Master Data Management & Bulk Upload)")
    tab1, tab2 = st.tabs(["🏢 分店清單 (Branches)", "🍱 餐點品項 (Menu Items)"])

    with tab1:
        st.markdown("#### 批次上傳分店清單 (Bulk Upload Branch Directory)")
        up_b = st.file_uploader("上傳分店 CSV (Upload Store CSV - Overwrites)", type=["csv"], key="up_b")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b)
                req_b = {"name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"}
                if req_b.issubset(new_df_b.columns):
                    new_df_b.to_csv(BRANCH_FILE, index=False)
                    st.success(f"🎉 成功更新 {len(new_df_b)} 間分店！(Successfully updated {len(new_df_b)} branches!)")
                    st.rerun()
                else: 
                    st.error(f"Missing required columns: {req_b}")
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### 線上手動編輯 (Live Branch Editor)")
        edit_b = st.data_editor(df_b, num_rows="dynamic", use_container_width=True, key="ed_b")
        if st.button("💾 儲存分店手動修改 (Save Branch Directory)", type="primary"):
            edit_b.to_csv(BRANCH_FILE, index=False)
            st.success("✅ 分店清單已成功儲存！(Branch directory saved successfully!)")
            st.rerun()

    with tab2:
        st.markdown("#### 批次上傳餐點清單 (Bulk Upload Menu Directory)")
        up_d = st.file_uploader("上傳餐點 CSV (Upload Menu CSV - Overwrites)", type=["csv"], key="up_d")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d)
                req_d = {"dish_id", "name", "main_carb", "protein"}
                if req_d.issubset(new_df_d.columns):
                    new_df_d.to_csv(DISH_FILE, index=False)
                    st.success(f"🎉 成功更新 {len(new_df_d)} 項餐點！(Successfully updated {len(new_df_d)} dishes!)")
                    st.rerun()
                else: 
                    st.error(f"Missing required columns: {req_d}")
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### 線上手動編輯 (Live Menu Editor)")
        edit_d = st.data_editor(df_d, num_rows="dynamic", use_container_width=True, key="ed_d")
        if st.button("💾 儲存餐點手動修改 (Save Menu Directory)", type="primary"):
            edit_d.to_csv(DISH_FILE, index=False)
            st.success("✅ 餐點清單已成功儲存！(Menu directory saved successfully!)")
            st.rerun()

# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    st.set_page_config(
        page_title="TrayZero | 大家樂智能餐盤審計系統", 
        page_icon="🍽️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    init_db()
    df_b, df_d = load_master_data()

    with st.spinner("🚀 正在啟動雙核心 AI 引擎 (Initializing AI Engines)..."):
        engine = load_ai_engine()

    render_header()

    # 側邊欄 Logo：透過 columns + CSS 強制絕對水平居中
    logo_path = "CDC_810.png" if os.path.exists("CDC_810.png") else ("CDC_810.jpg" if os.path.exists("CDC_810.jpg") else None)
    if logo_path:
        col_l1, col_l2, col_l3 = st.sidebar.columns([0.15, 0.7, 0.15])
        with col_l2:
            st.image(logo_path, width=175)
    else:
        st.sidebar.warning("⚠️ 請上傳 CDC_810.png 至根目錄")

    st.sidebar.title("🎛️ 系統控制台 (Control Panel)")
    mode = st.sidebar.radio("工作模式 (Navigation)", [
        "Mode 1: 前線餐盤智能偵測 (Live Tray Audit Station)",
        "Mode 2: 總部即時營運大盤 (Executive HQ Dashboard)",
        "Mode 3: 基礎資料設定 (Master Data Management)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("系統測試維護 (System Maintenance)")
    if st.sidebar.button("🗑️ 清空審計資料庫 (Reset Audit DB)", type="secondary"):
        reset_db()
        st.session_state["latest"] = None
        st.session_state["last_h"] = None
        st.session_state["last_uploaded_sig"] = None
        st.sidebar.success("✅ 資料庫已完全清空！(Database cleared!)")
        st.rerun()

    if mode.startswith("Mode 1"): 
        render_mode1(df_b, df_d, engine)
    elif mode.startswith("Mode 2"): 
        render_mode2(df_b, df_d, engine)
    else: 
        render_mode3(df_b, df_d)

if __name__ == "__main__":
    main()
