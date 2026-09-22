import os, datetime, sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================================================================
# 0. 系統檔案設定與白名單
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
# 1. 樣式注入：現代白色 Bento Grid + 紫色側邊欄
# ==============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        .stApp { background-color: #F4F6FB !important; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #4F46E5 0%, #3730A3 100%) !important;
            border-right: none !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
            color: #E0E7FF !important;
        }
        .trayzero-header {
            background: #FFFFFF; border-radius: 18px; padding: 18px 24px; margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.03); display: flex; align-items: center; border: 1px solid #E2E8F0;
        }
        .trayzero-title { color: #1E1B4B !important; font-size: 1.3rem; font-weight: 800; white-space: nowrap; margin: 0; }
        .bento-card {
            background: #FFFFFF; border-radius: 16px; padding: 18px 20px; margin-bottom: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03); border: 1px solid #E2E8F0;
        }
        .bento-label { font-size: 0.8rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
        .bento-val { font-size: 1.8rem; font-weight: 900; color: #0F172A; }
        .directive-card {
            border-radius: 14px; padding: 16px 18px; margin-bottom: 10px; background: #FFFFFF;
            border-left: 5px solid; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #F1F5F9;
        }
        .directive-chef { border-left: 5px solid #EF4444; }
        .directive-pos { border-left: 5px solid #F59E0B; }
        .directive-mgr { border-left: 5px solid #4F46E5; }
        .directive-title { font-weight: 800; font-size: 0.9rem; color: #1E293B; margin-bottom: 4px; }
        .directive-body { font-size: 0.85rem; color: #475569; line-height: 1.5; }
        .sidebar-logo-box {
            background: #FFFFFF; border-radius: 14px; padding: 10px 14px; display: flex;
            justify-content: center; margin-bottom: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. 資料庫與檔案 I/O
# ==============================================================================
def db_conn(): return sqlite3.connect(DB_FILE)

def init_db():
    with db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, audit_date TEXT, audit_month TEXT,
                branch_name TEXT, branch_level TEXT, dish_name TEXT, primary_waste TEXT,
                waste_ratio REAL, cost_waste_hkd REAL, co2_emission_kg REAL
            )
        """)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(audit_logs)").fetchall()]
        if "audit_date" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_date TEXT")
        if "audit_month" not in cols: conn.execute("ALTER TABLE audit_logs ADD COLUMN audit_month TEXT")

def save_record(r):
    with db_conn() as conn:
        conn.execute("""
            INSERT INTO audit_logs (timestamp, audit_date, audit_month, branch_name, branch_level,
            dish_name, primary_waste, waste_ratio, cost_waste_hkd, co2_emission_kg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (r["timestamp"], r["audit_date"], r["audit_month"], r["branch_name"], r["branch_level"],
              r["dish_name"], r["primary_waste"], r["waste_ratio"], r["cost_waste_hkd"], r["co2_emission_kg"]))

def get_records():
    with db_conn() as conn: return pd.read_sql("SELECT * FROM audit_logs ORDER BY id DESC", conn)

def reset_db():
    with db_conn() as conn: conn.execute("DELETE FROM audit_logs")

def load_master_data():
    df_b = pd.read_csv(BRANCH_FILE) if os.path.exists(BRANCH_FILE) else pd.DataFrame(columns=["name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"])
    df_d = pd.read_csv(DISH_FILE) if os.path.exists(DISH_FILE) else pd.DataFrame(columns=["dish_id", "name", "main_carb", "protein"])
    return df_b, df_d

# ==============================================================================
# 3. AI 模型引擎
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
    with torch.no_grad(): out = engine["det"](**inp)
    sz = torch.tensor([image.size[::-1]]).to(engine["device"])
    res = engine["proc"].post_process_object_detection(out, threshold=0.20, target_sizes=sz)[0]
    
    img_draw, total_a, waste_a = image.copy(), image.size[0] * image.size[1], 0
    draw = ImageDraw.Draw(img_draw)
    items, valid_food = [], False
    color_map = {"Rice": "#EF4444", "Meat": "#F59E0B", "Veg_Soup": "#10B981"}
    primary = "光盤 (Clean Plate)"

    for box, score, label_id in zip(res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()):
        lbl = engine["det"].config.id2label.get(label_id, "item")
        if lbl not in FOOD_WHITELIST: continue
        cat = FOOD_WHITELIST[lbl]
        valid_food = True
        
        if cat == "Rice": name, primary = "白飯殘留", "白飯殘留"
        elif cat == "Meat":
            name = "肉類殘留"
            if "白飯" not in primary: primary = "肉類殘留"
        elif cat == "Veg_Soup":
            name = f"配菜/醬汁 ({lbl})"
            if primary == "光盤 (Clean Plate)": primary = "配菜/醬汁"
        else: name = "餐盤基準"

        b = [max(0, box[0]), max(0, box[1]), min(image.size[0], box[2]), min(image.size[1], box[3])]
        area = (b[2] - b[0]) * (b[3] - b[1])
        if cat != "Tray": waste_a += area

        c = color_map.get(cat, "#4F46E5")
        draw.rectangle(b, outline=c, width=3)
        draw.text((b[0]+3, b[1]+3), f"{name} {score:.0%}", fill=c)
        items.append({"類別": name, "置信度": f"{score:.1%}", "佔比": f"{area/total_a:.1%}"})

    ratio = min(1.0, waste_a / (total_a * 0.65)) if (total_a > 0 and valid_food) else 0.0
    return img_draw, items, ratio, primary, valid_food

def auto_detect_dish(image, candidate_dishes):
    if not candidate_dishes: return "未定義餐點", 0.0
    np_img = np.array(image.resize((32, 32)))
    r, g, b = np.mean(np_img[:, :, 0]), np.mean(np_img[:, :, 1]), np.mean(np_img[:, :, 2])
    idx = 1 if (r > 140 and g > 110 and b < 90 and len(candidate_dishes) > 1) else 0
    return candidate_dishes[idx], 0.88

# ==============================================================================
# 4. 宏觀建議 (Mode 2 Advisory Hub)
# ==============================================================================
def get_advisory(df, scope_type, branch_sel, dish_sel, engine):
    if df.empty:
        n, avg_w, loss, co2 = 0, 0.0, 0.0, 0.0
        t_branch = branch_sel if branch_sel != "ALL" else "全港分店"
        t_dish = dish_sel if dish_sel != "ALL" else "全部餐點"
    else:
        n = len(df)
        avg_w = df["waste_ratio"].mean()
        loss = df["cost_waste_hkd"].sum()
        co2 = df["co2_emission_kg"].sum()
        t_branch = branch_sel if branch_sel != "ALL" else df.groupby("branch_name")["waste_ratio"].mean().idxmax()
        t_dish = dish_sel if dish_sel != "ALL" else df.groupby("dish_name")["waste_ratio"].mean().idxmax()

    actions = [
        {"type": "directive-chef", "role": f"👨‍🍳 後廚出餐負責人 ({t_branch} - {t_dish})",
         "text": f"全期殘食率 {avg_w:.1f}%。即刻針對「{t_dish}」更換為 3 號標準平底飯勺（減量 30g 出餐），嚴控備餐積壓。"},
        {"type": "directive-pos", "role": "🖥️ 門市 POS / Kiosk 運營",
         "text": f"於「{t_branch}」自助點餐機置頂彈窗提示「少飯減扣 $2」優惠，引流低食量顧客選擇輕量裝。"},
        {"type": "directive-mgr", "role": "📦 門市經理與採購部",
         "text": f"下調次輪電飯煲蒸煮量 10%，預計防損挽回 HK$ {max(150, round(loss * 0.4)):,.0f}，月減碳 {co2:.1f} kg CO2e。"}
    ]
    try:
        p = f"You are CEO of Cafe de Coral. Review: {n} trays, avg waste {avg_w:.1f}%, loss HK${loss:.0f}, target {t_branch}, dish {t_dish}. Write 1 short strategic order."
        inp = engine["tok"](p, return_tensors="pt", max_length=256, truncation=True).to(engine["device"])
        memo = engine["tok"].decode(engine["gen"].generate(**inp, max_new_tokens=60)[0], skip_special_tokens=True)
    except Exception:
        memo = f"核准：實施 {scope_type} 殘食校準方針，精準優化各門市配給。"

    return {"total": n, "avg_w": avg_w, "branch": t_branch, "dish": t_dish, "actions": actions, "memo": memo}

# ==============================================================================
# 5. UI 各模式渲染
# ==============================================================================
def render_header():
    st.markdown("""
    <div class="trayzero-header">
        <h2 class="trayzero-title">🍽️ TrayZero Intelligent plate waste auditing and central distribution system</h2>
    </div>
    """, unsafe_allow_html=True)

def render_mode1(df_b, df_d, engine):
    if df_b.empty or df_d.empty:
        st.warning("⚠️ 門市或餐點清單為空！請先切換至「Mode 3: 基礎資料設定」上傳 (Bulk Upload) CSV。")
        return

    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("#### 🏢 門市與掃描設定")
        b_name = st.selectbox("執勤門市", df_b["name"].tolist())
        b_meta = df_b[df_b["name"] == b_name].iloc[0]
        st.caption(f"等級: `{b_meta['level']}` | 區域: `{b_meta['district']}` | 標配飯量: `{b_meta['base_rice_g']}g`")

        auto_dish = st.checkbox("🤖 AI 自動辨識餐點類型 (Auto-Detect Dish)", value=True)
        scan_mode = st.radio("掃描模式", ["🟢 Live Camera 靜止感應", "📸 手動快照", "📁 上傳照片"], horizontal=True)
        
        img_cap, do_scan = None, False
        if scan_mode == "🟢 Live Camera 靜止感應":
            cam = st.camera_input("持續監控畫面", key="live_cam")
            if cam:
                img_cap = Image.open(cam).convert("RGB")
                h = hash(img_cap.tobytes()[:3000])
                if h != st.session_state.get("last_h"):
                    st.session_state["last_h"], do_scan = h, True
                else: st.info("🟢 監控中：當前餐盤已完成分析，等待更換餐盤...")
        elif scan_mode == "📸 手動快照":
            m_cam = st.camera_input("拍照", key="manual_cam")
            if m_cam: img_cap, do_scan = Image.open(m_cam).convert("RGB"), True
        else:
            up = st.file_uploader("上傳餐盤相片", type=["jpg", "png", "jpeg"])
            if up:
                img_cap = Image.open(up).convert("RGB")
                if st.button("🚀 執行上傳分析", type="primary"): do_scan = True

        if img_cap and do_scan:
            with st.spinner("AI 偵測中: YOLOS 正在過濾非餐盤目標並辨識殘食..."):
                anno_img, items, ratio, primary_cat, is_food = detect_tray(img_cap, engine)

            if not is_food:
                st.error("🚫 偵測失敗：未檢測到合法餐盤或食物物件！（已自動過濾人物/背景）")
                st.session_state["latest"] = None
            else:
                sel_dish, _ = auto_detect_dish(img_cap, df_d["name"].tolist()) if auto_dish else (st.selectbox("指定餐點", df_d["name"].tolist()), 1.0)
                loss_hkd = round(ratio * 25 * 0.45, 1)
                now = datetime.datetime.now()
                save_record({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "audit_date": now.strftime("%Y-%m-%d"),
                    "audit_month": now.strftime("%Y-%m"), "branch_name": b_name, "branch_level": str(b_meta['level']).split(" ")[0],
                    "dish_name": sel_dish, "primary_waste": primary_cat, "waste_ratio": round(ratio * 100, 1),
                    "cost_waste_hkd": loss_hkd, "co2_emission_kg": round(loss_hkd * 0.12, 2)
                })
                st.session_state["latest"] = {
                    "img": anno_img, "dish": sel_dish, "time": now.strftime("%H:%M:%S"),
                    "ratio": ratio, "cat": primary_cat, "cost": loss_hkd, "branch": b_name, "items": items
                }
                st.toast("✅ 掃描成功並寫入資料庫！")

    with c2:
        st.markdown("#### 🎯 前線掃描結果 (Latest Result)")
        latest = st.session_state.get("latest")
        if not latest:
            st.info("💡 尚未執行偵測或畫面非餐盤。請對準餐盤掃描。")
        else:
            st.image(latest["img"], caption=f"{latest['dish']} ({latest['time']})", use_container_width=True)
            k1, k2, k3 = st.columns(3)
            k1.markdown(f'<div class="bento-card"><div class="bento-label">殘食佔比</div><div class="bento-val" style="color:#EF4444">{latest["ratio"]:.1%}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="bento-card"><div class="bento-label">主要殘留</div><div class="bento-val" style="font-size:1.15rem;margin-top:6px;">{latest["cat"]}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="bento-card"><div class="bento-label">損耗金額</div><div class="bento-val" style="color:#F59E0B">HK${latest["cost"]}</div></div>', unsafe_allow_html=True)
            st.success(f"📥 **記錄已歸檔**：已綁定至 `{latest['branch']}`。")

def render_mode2(df_b, df_d, engine):
    st.markdown("### 📊 總部即時營運大盤 & 戰略建議")
    df_raw = get_records()

    st.markdown("#### 🎛️ 雙軸分析維度 (Dimensions)")
    c1, c2 = st.columns(2)
    with c1:
        b_filter = st.selectbox("1. 門市維度", ["🌐 全部分店 (Overall)"] + df_b["name"].tolist() if not df_b.empty else ["🌐 全部分店 (Overall)"])
        sel_b = "ALL" if "全部" in b_filter else b_filter
    with c2:
        d_filter = st.selectbox("2. 食物種類維度", ["🍱 全部餐點 (Overall)"] + df_d["name"].tolist() if not df_d.empty else ["🍱 全部餐點 (Overall)"])
        sel_d = "ALL" if "全部" in d_filter else d_filter

    df_filtered = df_raw.copy()
    if not df_filtered.empty:
        if sel_b != "ALL": df_filtered = df_filtered[df_filtered["branch_name"] == sel_b]
        if sel_d != "ALL": df_filtered = df_filtered[df_filtered["dish_name"] == sel_d]

    # KPI 統計
    n = len(df_filtered)
    avg_w = df_filtered["waste_ratio"].mean() if n > 0 else 0.0
    tot_hkd = df_filtered["cost_waste_hkd"].sum() if n > 0 else 0.0
    tot_co2 = df_filtered["co2_emission_kg"].sum() if n > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="bento-card"><div class="bento-label">樣本盤數</div><div class="bento-val">{n} <span style="font-size:0.9rem;color:#94A3B8">TRAYS</span></div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="bento-card"><div class="bento-label">平均殘食率</div><div class="bento-val" style="color:#EF4444">{avg_w:.1f}%</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="bento-card"><div class="bento-label">食材損耗總額</div><div class="bento-val" style="color:#F59E0B">HK${tot_hkd:,.1f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="bento-card"><div class="bento-label">累計碳排放</div><div class="bento-val" style="color:#4F46E5">{tot_co2:.2f} <span style="font-size:0.9rem;color:#94A3B8">kg</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    scope = st.radio("覆盤時限", ["📅 日度營運建議 (Daily Review)", "🗓️ 月度戰略採購建議 (Monthly Advisory)"], horizontal=True)
    scope_code = "DAILY" if "日度" in scope else "MONTHLY"
    date_col = "audit_date" if scope_code == "DAILY" else "audit_month"
    
    dates = df_filtered[date_col].dropna().unique().tolist() if not df_filtered.empty else [datetime.date.today().strftime("%Y-%m-%d" if scope_code=="DAILY" else "%Y-%m")]
    s_date = st.selectbox(f"選擇審計{'日期' if scope_code=='DAILY' else '月份'}", dates)
    df_scope = df_filtered[df_filtered[date_col] == s_date] if not df_filtered.empty else pd.DataFrame()

    with st.spinner("AI 正在分析生成營運指引..."):
        adv = get_advisory(df_scope, scope_code, sel_b, sel_d, engine)

    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        st.markdown(f"""
        <div class="bento-card">
            <div class="bento-label">當期指標摘要</div>
            <div style="font-size:0.9rem;color:#334155;line-height:1.8;margin-top:8px;">
                • 審計盤數: <b>{adv['total']} 盤</b><br>• 殘食率: <b style="color:#EF4444">{adv['avg_w']:.1f}%</b><br>
                • 目標門市: <b>{adv['branch']}</b><br>• 目標餐點: <b>{adv['dish']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        for a in adv["actions"]:
            st.markdown(f'<div class="directive-card {a["type"]}"><div class="directive-title">{a["role"]}</div><div class="directive-body">{a["text"]}</div></div>', unsafe_allow_html=True)
        with st.expander("📝 檢視 AI 總監決策備忘錄 (Executive Memo)", expanded=True):
            st.write(adv["memo"])

    st.markdown("---")
    if not df_filtered.empty:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### 🏢 各門市平均殘食率 (%)")
            st.bar_chart(df_filtered.groupby("branch_name")["waste_ratio"].mean(), color="#4F46E5")
        with g2:
            st.markdown("##### 🍱 各食物種類耗損 (HK$)")
            st.bar_chart(df_filtered.groupby("dish_name")["cost_waste_hkd"].sum(), color="#EF4444")
        st.dataframe(df_filtered, use_container_width=True)
    else:
        st.info("💡 目前所選維度尚無過盤紀錄。")

def render_mode3(df_b, df_d):
    st.markdown("### ⚙️ 基礎資料管理 (Bulk Upload & Master Data)")
    tab1, tab2 = st.tabs(["🏢 分店清單 (Branches)", "🍱 餐點品項 (Dishes)"])

    with tab1:
        st.markdown("#### 批次上傳分店清單 (Bulk Upload)")
        up_b = st.file_uploader("上傳分店 CSV (覆蓋更新)", type=["csv"], key="up_b")
        if up_b:
            try:
                new_df_b = pd.read_csv(up_b)
                req_b = {"name", "level", "district", "traffic", "avg_covers", "base_rice_g", "strategy"}
                if req_b.issubset(new_df_b.columns):
                    new_df_b.to_csv(BRANCH_FILE, index=False)
                    st.success(f"🎉 成功更新 {len(new_df_b)} 間分店！")
                    st.rerun()
                else: st.error(f"欄位必須包含: {req_b}")
            except Exception as e: st.error(f"上傳錯誤: {e}")

        st.markdown("#### 線上手動編輯")
        edit_b = st.data_editor(df_b, num_rows="dynamic", use_container_width=True, key="ed_b")
        if st.button("💾 儲存分店手動修改", type="primary"):
            edit_b.to_csv(BRANCH_FILE, index=False)
            st.success("✅ 分店已儲存！")
            st.rerun()

    with tab2:
        st.markdown("#### 批次上傳餐點清單 (Bulk Upload)")
        up_d = st.file_uploader("上傳餐點 CSV (覆蓋更新)", type=["csv"], key="up_d")
        if up_d:
            try:
                new_df_d = pd.read_csv(up_d)
                req_d = {"dish_id", "name", "main_carb", "protein"}
                if req_d.issubset(new_df_d.columns):
                    new_df_d.to_csv(DISH_FILE, index=False)
                    st.success(f"🎉 成功更新 {len(new_df_d)} 項餐點！")
                    st.rerun()
                else: st.error(f"欄位必須包含: {req_d}")
            except Exception as e: st.error(f"上傳錯誤: {e}")

        st.markdown("#### 線上手動編輯")
        edit_d = st.data_editor(df_d, num_rows="dynamic", use_container_width=True, key="ed_d")
        if st.button("💾 儲存餐點手動修改", type="primary"):
            edit_d.to_csv(DISH_FILE, index=False)
            st.success("✅ 餐點已儲存！")
            st.rerun()

# ==============================================================================
# 6. 主程序入口
# ==============================================================================
def main():
    st.set_page_config(page_title="TrayZero | 大家樂智能審計", page_icon="🍽️", layout="wide")
    inject_custom_css()
    init_db()
    df_b, df_d = load_master_data()

    with st.spinner("🚀 正在啟動 AI 引擎..."):
        engine = load_ai_engine()

    render_header()

    # 側邊欄 Logo：置中並縮小 30% (width=180)
    st.sidebar.markdown('<div class="sidebar-logo-box">', unsafe_allow_html=True)
    if os.path.exists("CDC_810.png"): st.sidebar.image("CDC_810.png", width=180)
    elif os.path.exists("CDC_810.jpg"): st.sidebar.image("CDC_810.jpg", width=180)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.title("🎛️ 系統控制台")
    mode = st.sidebar.radio("工作模式", [
        "Mode 1: 前線餐盤智能偵測 (Tray Station)",
        "Mode 2: 總部即時營運大盤 (HQ Dashboard)",
        "Mode 3: 基礎資料設定 (Master Data)"
    ])
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ 清空審計資料庫 (Reset DB)", type="secondary"):
        reset_db()
        st.session_state["latest"] = None
        st.session_state["last_h"] = None
        st.sidebar.success("✅ 資料庫已完全清空！")
        st.rerun()

    if mode.startswith("Mode 1"): render_mode1(df_b, df_d, engine)
    elif mode.startswith("Mode 2"): render_mode2(df_b, df_d, engine)
    else: render_mode3(df_b, df_d)

if __name__ == "__main__":
    main()
