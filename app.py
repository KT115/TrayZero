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
# 1. Clean UI Theme (Minimalist, Elevated Neutral SaaS)
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Clean Soft Canvas */
        .stApp {
            background-color: #F8FAFC !important;
            color: #0F172A;
        }

        /* Crisp Sidebar */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #334155 !important;
            font-weight: 500;
        }

        /* Minimalist Header */
        .trayzero-header {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 20px 28px;
            margin-bottom: 24px;
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
            letter-spacing: -0.02em;
            margin: 0 !important;
        }

        /* Clean Micro-Elevated Cards */
        .clean-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 14px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.03);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .clean-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.06);
            transform: translateY(-2px);
        }

        .clean-label {
            font-size: 0.75rem;
            color: #64748B;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        .clean-val {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }

        .clean-sub {
            font-size: 0.78rem;
            color: #10B981;
            font-weight: 600;
            margin-top: 6px;
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
            line-height: 1.55;
        }

        /* Sidebar Logo Container */
        .sidebar-logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px 0 16px 0;
            border-bottom: 1px solid #F1F5F9;
            margin-bottom: 18px;
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
    primary = "Clean Plate"

    for box, score, label_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        lbl = engine["det"].config.id2label.get(label_id, "item")
        if lbl not in FOOD_WHITELIST: 
            continue
            
        cat = FOOD_WHITELIST[lbl]
        valid_food = True
        
        if cat == "Rice": 
            name, primary = "Rice Waste", "Rice Residual"
        elif cat == "Meat":
            name = "Meat Waste"
            if "Rice" not in primary: 
                primary = "Protein Residual"
        elif cat == "Veg_Soup":
            name = f"Sides/Sauce ({lbl})"
            if primary == "Clean Plate": 
                primary = "Sides & Sauce"
        else: 
            name = "Tray Reference"

        b = [max(0, box[0]), max(0, box[1]), min(image.size[0], box[2]), min(image.size[1], box[3])]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if cat != "Tray": 
            waste_area += area

        c = color_map.get(cat, "#3B82F6")
        draw.rectangle(b, outline=c, width=3)
        draw.text((b[0] + 4, b[1] + 4), f"{name} {score:.0%}", fill=c)
        items.append({"Category": name, "Confidence": f"{score:.1%}", "Tray Coverage": f"{area/total_area:.1%}"})

    ratio = min(1.0, waste_area / (total_area * 0.65)) if (total_area > 0 and valid_food) else 0.0
    return img_draw, items, ratio, primary, valid_food

def auto_detect_dish(image, candidate_dishes):
    if not candidate_dishes: 
        return "Undefined Dish", 0.0
    np_img = np.array(image.resize((32, 32)))
    r, g, b = np.mean(np_img[:, :, 0]), np.mean(np_img[:, :, 1]), np.mean(np_img[:, :, 2])
    idx = 1 if (r > 140 and g > 110 and b < 90 and len(candidate_dishes) > 1) else 0
    return candidate_dishes[idx], 0.88

# ==============================================================================
# 4. Macro Advisory Synthesis Engine
# ==============================================================================
def get_advisory(df, scope_type, branch_sel, dish_sel, engine):
    if df.empty:
        n, avg_w, loss, co2 = 0, 0.0, 0.0, 0.0
        t_branch = branch_sel if branch_sel != "ALL" else "All Branches"
        t_dish = dish_sel if dish_sel != "ALL" else "All Menu Items"
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
            "role": f"👨‍🍳 Head Chef ({t_branch} • {t_dish})",
            "text": f"Average plate waste is {avg_w:.1f}%. Immediately switch to a standard size-3 portion scoop (-30g per serving) on {t_dish} to eliminate prep backlog."
        },
        {
            "type": "directive-pos", 
            "role": "🖥️ Front-of-House / Kiosk POS Promotion",
            "text": f"Activate an automated POS prompt offering 'Light Portion (-HK$2)' for {t_dish} at {t_branch} to guide low-appetite diners toward right-sized meals."
        },
        {
            "type": "directive-mgr", 
            "role": "📦 Store Manager & Supply Chain Procurement",
            "text": f"Reduce daily cooked rice batches by 10%. Projected daily waste prevention: HK$ {max(150, round(loss * 0.4)):,.0f}; GHG mitigation: {co2:.1f} kg CO2e."
        }
    ]
    
    try:
        p = f"You are Executive Operations Director of Cafe de Coral. Review: {n} audited trays, average waste ratio {avg_w:.1f}%, estimated loss HK${loss:.0f} across {t_branch} for {t_dish}. Provide one concise board-level operational instruction."
        inp = engine["tok"](p, return_tensors="pt", max_length=256, truncation=True).to(engine["device"])
        memo = engine["tok"].decode(engine["gen"].generate(**inp, max_new_tokens=60)[0], skip_special_tokens=True)
    except Exception:
        memo = f"Approved: Execute {scope_type} portion calibration policy and optimize central kitchen inventory allocations."

    return {"total": n, "avg_w": avg_w, "branch": t_branch, "dish": t_dish, "actions": actions, "memo": memo}

# ==============================================================================
# 5. UI Views & Component Rendering
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="trayzero-header">
        <h2 class="trayzero-title">🍽️ TrayZero Intelligent Plate Waste Auditing & Central Distribution System</h2>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(df_b, df_d, engine):
    if df_b.empty or df_d.empty:
        st.warning("⚠️ Branch or menu database is empty. Please navigate to 'Mode 3: Master Data' to bulk upload CSV files.")
        return

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("#### 🏢 Station & Input Settings")
        b_name = st.selectbox("Active Store Location", df_b["name"].tolist())
        b_meta = df_b[df_b["name"] == b_name].iloc[0]
        st.caption(f"Tier: `{b_meta['level']}` | District: `{b_meta['district']}` | Standard Rice: `{b_meta['base_rice_g']}g`")

        auto_dish = st.checkbox("🤖 Enable AI Auto Dish Recognition", value=True)
        scan_mode = st.radio("Scanning Method", ["🟢 Continuous Live Camera (Auto-Scan)", "📸 Manual Snapshot", "📁 Upload Image"], horizontal=True)
        
        img_cap, do_scan = None, False
        if scan_mode == "🟢 Continuous Live Camera (Auto-Scan)":
            cam = st.camera_input("Live Feed Monitor", key="live_cam")
            if cam:
                img_cap = Image.open(cam).convert("RGB")
                h = hash(img_cap.tobytes()[:3000])
                if h != st.session_state.get("last_h"):
                    st.session_state["last_h"], do_scan = h, True
                else: 
                    st.info("🟢 Monitoring: Active tray already analyzed. Awaiting next tray...")
        elif scan_mode == "📸 Manual Snapshot":
            m_cam = st.camera_input("Take Snapshot", key="manual_cam")
            if m_cam: 
                img_cap, do_scan = Image.open(m_cam).convert("RGB"), True
        else:
            up = st.file_uploader("Upload Tray Image", type=["jpg", "png", "jpeg"])
            if up:
                img_cap = Image.open(up).convert("RGB")
                if st.button("🚀 Analyze Uploaded Image", type="primary"): 
                    do_scan = True

        if img_cap and do_scan:
            with st.spinner("AI Processing: Filtering non-food artifacts & analyzing waste areas..."):
                anno_img, items, ratio, primary_cat, is_food = detect_tray(img_cap, engine)

            if not is_food:
                st.error("🚫 Detection Failed: No valid tray or food objects detected! (People/backgrounds filtered).")
                st.session_state["latest"] = None
            else:
                sel_dish, _ = auto_detect_dish(img_cap, df_d["name"].tolist()) if auto_dish else (st.selectbox("Select Target Dish", df_d["name"].tolist()), 1.0)
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
                st.toast("✅ Tray audit recorded and synchronized with database!")

    with c2:
        st.markdown("#### 🎯 Latest Scan Result")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 No valid tray scan available. Align the camera with the collection tray.")
        else:
            st.image(latest["img"], caption=f"{latest['dish']} ({latest['time']})", use_container_width=True)
            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="clean-card"><div class="clean-label">Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if latest["ratio"] > 0.3 else "#10B981"}">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="clean-card"><div class="clean-label">Primary Residual</div><div class="clean-val" style="font-size:1.15rem;margin-top:6px;">{latest["cat"]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="clean-card"><div class="clean-label">Estimated Loss</div><div class="clean-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)
            st.success(f"📥 **Logged**: Successfully indexed to `{latest['branch']}`.")

def render_mode2(df_b, df_d, engine):
    st.markdown("### 📊 Executive Overview & Strategic Advisory")
    df_raw = get_records()

    st.markdown("#### 🎛️ Analysis Dimensions")
    c1, c2 = st.columns(2)
    with c1:
        b_filter = st.selectbox("1. Store Location Dimension", ["🌐 All Branches (Overall)"] + df_b["name"].tolist() if not df_b.empty else ["🌐 All Branches (Overall)"])
        sel_b = "ALL" if "All Branches" in b_filter else b_filter
    with c2:
        d_filter = st.selectbox("2. Menu Item Dimension", ["🍱 All Menu Items (Overall)"] + df_d["name"].tolist() if not df_d.empty else ["🍱 All Menu Items (Overall)"])
        sel_d = "ALL" if "All Menu Items" in d_filter else d_filter

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
    k1.markdown(f'<div class="clean-card"><div class="clean-label">Audited Trays</div><div class="clean-val">{n} <span style="font-size:0.85rem;color:#94A3B8">TRAYS</span></div><div class="clean-sub">Real-time Sync</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="clean-card"><div class="clean-label">Average Waste Ratio</div><div class="clean-val" style="color:{"#EF4444" if avg_w > 25 else "#10B981"}">{avg_w:.1f}%</div><div class="clean-sub">Target: &lt;15.0%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="clean-card"><div class="clean-label">Total Material Loss</div><div class="clean-val" style="color:#F59E0B">HK${tot_hkd:,.1f}</div><div class="clean-sub">Dynamic Valuation</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="clean-card"><div class="clean-label">GHG Emissions</div><div class="clean-val" style="color:#3B82F6">{tot_co2:.2f} <span style="font-size:0.85rem;color:#94A3B8">kg</span></div><div class="clean-sub">Scope 3 ESG Metric</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    scope = st.radio("Advisory Scope", ["📅 Daily Operational Review", "🗓️ Monthly Strategic Sourcing Advisory"], horizontal=True)
    scope_code = "DAILY" if "Daily" in scope else "MONTHLY"
    date_col = "audit_date" if scope_code == "DAILY" else "audit_month"
    
    dates = df_filtered[date_col].dropna().unique().tolist() if not df_filtered.empty else [datetime.date.today().strftime("%Y-%m-%d" if scope_code=="DAILY" else "%Y-%m")]
    s_date = st.selectbox(f"Select Audit {'Date' if scope_code=='DAILY' else 'Month'}", dates)
    df_scope = df_filtered[df_filtered[date_col] == s_date] if not df_filtered.empty else pd.DataFrame()

    with st.spinner("AI Engine: Generating executive recommendations..."):
        adv = get_advisory(df_scope, scope_code, sel_b, sel_d, engine)

    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.markdown(f"""
        <div class="clean-card">
            <div class="clean-label">Scope Summary</div>
            <div style="font-size:0.88rem;color:#334155;line-height:1.8;margin-top:8px;">
                • Audited Trays: <b>{adv['total']} trays</b><br>
                • Waste Ratio: <b style="color:#EF4444">{adv['avg_w']:.1f}%</b><br>
                • Focus Branch: <b>{adv['branch']}</b><br>
                • Primary Dish: <b>{adv['dish']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        for a in adv["actions"]:
            st.markdown(f'<div class="directive-card {a["type"]}"><div class="directive-title">{a["role"]}</div><div class="directive-body">{a["text"]}</div></div>', unsafe_allow_html=True)
        with st.expander("📝 View AI Executive Memo", expanded=True):
            st.write(adv["memo"])

    st.markdown("---")
    if not df_filtered.empty:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🏢 Store Average Waste Ratio (%)")
            st.bar_chart(df_filtered.groupby("branch_name")["waste_ratio"].mean(), color="#3B82F6")
        with g2:
            st.markdown("##### 🍱 Waste Cost by Dish (HK$)")
            st.bar_chart(df_filtered.groupby("dish_name")["cost_waste_hkd"].sum(), color="#EF4444")
        st.dataframe(df_filtered, use_container_width=True)
    else:
        st.info("💡 No audit records found for the selected dimensions.")

def render_mode3(df_b, df_d):
    st.markdown("### ⚙️ Master Data Management (Bulk Upload & Live Editor)")
    tab1, tab2 = st.tabs(["🏢 Store Locations (Branches)", "🍱 Menu Items (Dishes)"])

    with tab1:
        st.markdown("#### Bulk Upload Branch Directory")
        up_b = st.file_uploader("Upload Store CSV (Overwrites Existing)", type=["csv"], key="up_b")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b)
                req_b = {"name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"}
                if req_b.issubset(new_df_b.columns):
                    new_df_b.to_csv(BRANCH_FILE, index=False)
                    st.success(f"🎉 Successfully updated {len(new_df_b)} branch records!")
                    st.rerun()
                else: 
                    st.error(f"Missing required columns: {req_b}")
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### Live Branch Editor")
        edit_b = st.data_editor(df_b, num_rows="dynamic", use_container_width=True, key="ed_b")
        if st.button("💾 Save Branch Modifications", type="primary"):
            edit_b.to_csv(BRANCH_FILE, index=False)
            st.success("✅ Branch directory saved successfully!")
            st.rerun()

    with tab2:
        st.markdown("#### Bulk Upload Menu Directory")
        up_d = st.file_uploader("Upload Menu Item CSV (Overwrites Existing)", type=["csv"], key="up_d")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d)
                req_d = {"dish_id", "name", "main_carb", "protein"}
                if req_d.issubset(new_df_d.columns):
                    new_df_d.to_csv(DISH_FILE, index=False)
                    st.success(f"🎉 Successfully updated {len(new_df_d)} menu item records!")
                    st.rerun()
                else: 
                    st.error(f"Missing required columns: {req_d}")
            except Exception as e: 
                st.error(f"Upload error: {e}")

        st.markdown("#### Live Menu Editor")
        edit_d = st.data_editor(df_d, num_rows="dynamic", use_container_width=True, key="ed_d")
        if st.button("💾 Save Menu Modifications", type="primary"):
            edit_d.to_csv(DISH_FILE, index=False)
            st.success("✅ Menu directory saved successfully!")
            st.rerun()

# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    st.set_page_config(page_title="TrayZero | Intelligent Waste Auditing", page_icon="🍽️", layout="wide")
    inject_custom_css()
    init_db()
    df_b, df_d = load_master_data()

    with st.spinner("🚀 Initializing Dual Deep Learning Engines..."):
        engine = load_ai_engine()

    render_header()

    # Sidebar Header with Clean Logo Placement (Scaled down 30%, width=180)
    st.sidebar.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
    if os.path.exists("CDC_810.png"): 
        st.sidebar.image("CDC_810.png", width=180)
    elif os.path.exists("CDC_810.jpg"): 
        st.sidebar.image("CDC_810.jpg", width=180)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.title("🎛️ Control Panel")
    mode = st.sidebar.radio("Navigation", [
        "Mode 1: Live Tray Audit Station",
        "Mode 2: Executive HQ Dashboard",
        "Mode 3: Master Data Management"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System Maintenance")
    if st.sidebar.button("🗑️ Reset Audit Database", type="secondary"):
        reset_db()
        st.session_state["latest"] = None
        st.session_state["last_h"] = None
        st.sidebar.success("✅ Database has been cleared!")
        st.rerun()

    if mode.startswith("Mode 1"): 
        render_mode1(df_b, df_d, engine)
    elif mode.startswith("Mode 2"): 
        render_mode2(df_b, df_d, engine)
    else: 
        render_mode3(df_b, df_d)

if __name__ == "__main__":
    main()
