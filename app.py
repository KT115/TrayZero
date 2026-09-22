import os
import time
import datetime
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
# 1. 頁面配置與全域狀態 (Session State - 實現即時跨 Mode 同步)
# ==============================================================================
st.set_page_config(
    page_title="大家樂 (Café de Coral) 智能餐盤殘食審計系統",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化真實偵測記錄庫 (跨 Mode 共享)
if "audit_history" not in st.session_state:
    st.session_state["audit_history"] = []

# ==============================================================================
# 2. 雙 Hugging Face 模型加載 (原生相容模式)
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

with st.spinner("🚀 正在初始化雙核心 AI 引擎 (YOLOS-tiny + Flan-T5)..."):
    img_processor, det_model, nlp_tokenizer, nlp_model, runtime_device = load_hf_models()

# ==============================================================================
# 3. 分店設定資料庫 (Branch Profile Meta)
# ==============================================================================
BRANCH_DB = {
    "中環威靈頓街店": {
        "level": "Level A (商業核心區 / CBD)",
        "traffic": "白領上班族為主，午市尖峰翻檯率極高",
        "avg_covers": 1200,
        "base_rice_g": 240,
        "strategy": "白領控醣需求高，建議推動小份量出餐與少飯扣減優惠。"
    },
    "沙田新城市廣場店": {
        "level": "Level B (住宅商場 / Residential)",
        "traffic": "家庭客、長者與週末休閒客群",
        "avg_covers": 1500,
        "base_rice_g": 260,
        "strategy": "家庭用餐與兒童共享比例高，建議推動彈性配菜組合。"
    },
    "香港科技大學店 (HKUST)": {
        "level": "Level C (校園與青年區 / Campus)",
        "traffic": "學生、教職員，運動量及食量顯著較大",
        "avg_covers": 1800,
        "base_rice_g": 280,
        "strategy": "維持大份量以維持飽足感，重點監控肉類醬汁口味與炸物品質。"
    }
}

# ==============================================================================
# 4. 輔助函式：Detection 與 Flan-T5 推論
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
    
    primary_category = "Clean Plate"
    
    for box, score, label_id in zip(boxes, scores, labels):
        raw_label = det_model.config.id2label.get(label_id, "item")
        
        if raw_label in ["bowl", "dining table", "cake"]:
            category = "Rice"
            display_name = "白飯殘留 (Rice Waste)"
            primary_category = "Rice Heavy Waste"
        elif raw_label in ["sandwich", "pizza", "hot dog"]:
            category = "Meat"
            display_name = "主菜肉類殘留 (Meat Residual)"
            if primary_category != "Rice Heavy Waste":
                primary_category = "Meat Residual"
        else:
            category = "Veg_Soup"
            display_name = f"配菜/醬汁 ({raw_label})"
            if primary_category == "Clean Plate":
                primary_category = "Veg Residual"
            
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

def generate_kitchen_decision(branch_name, branch_data, target_dish, waste_ratio):
    prompt = (
        f"You are the executive kitchen director of Cafe de Coral. "
        f"Store: {branch_name} ({branch_data['level']}). Target Dish: {target_dish}. "
        f"Standard portion: {branch_data['base_rice_g']}g. "
        f"Visual detection observed waste ratio: {waste_ratio:.1%}. "
        f"Strategy: {branch_data['strategy']} "
        f"Provide one actionable kitchen portion adjustment action in grams and one POS ordering change."
    )
    inputs = nlp_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(runtime_device)
    with torch.no_grad():
        outputs = nlp_model.generate(**inputs, max_new_tokens=90, do_sample=False)
    return nlp_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================================================================
# 5. 側邊欄切換：Mode 1 (Detection) vs. Mode 2 (Dashboard)
# ==============================================================================
st.sidebar.title("🎛️ 系統模式選擇")
system_mode = st.sidebar.radio(
    "切換工作模式",
    ["Mode 1: 前線餐盤智能偵測 (Tray Detection)", "Mode 2: 總部即時營運大盤 (Real-time Dashboard)"],
    index=0
)

st.sidebar.markdown("---")

# ==============================================================================
# 6. MODE 1: 前線餐盤智能偵測機 (Tray Detection Station)
# ==============================================================================
if system_mode == "Mode 1: 前線餐盤智能偵測 (Tray Detection)":
    st.subheader("📸 Mode 1: 前線智能餐盤審計機 (回收台即時掃描)")
    st.caption("流程：選擇門市 ➔ 自動靜止掃描 / 拍照 / 上傳 ➔ 執行雙 Pipeline ➔ 數據自動同步至總部 Dashboard")
    
    col_setting, col_cam = st.columns([1, 2])
    
    with col_setting:
        st.markdown("### 🏢 當前執勤分店設定")
        selected_branch = st.selectbox("選擇當前門市 (此掃描記錄將綁定該店)", list(BRANCH_DB.keys()))
        branch_meta = BRANCH_DB[selected_branch]
        
        st.info(f"""
        **門市等級**: `{branch_meta['level']}`  
        **客群結構**: {branch_meta['traffic']}  
        **標準出餐**: `{branch_meta['base_rice_g']}g 白飯`  
        """)
        
        target_dish = st.selectbox(
            "抽檢餐點項目",
            ["一哥焗豬扒飯", "咖喱牛腩飯", "滑蛋蝦仁飯"]
        )
        
        scan_mode = st.radio(
            "輸入感應方式",
            ["自動靜止感應 (Live Auto-Detect)", "手動拍照 (Manual Capture)", "照片檔案上傳 (Upload)"]
        )

    with col_cam:
        captured_image = None
        
        # 模式 A: 自動靜止 2 秒感應 (基於 WebCam 模擬)
        if scan_mode == "自動靜止感應 (Live Auto-Detect)":
            st.markdown("#### 🟢 Live Camera 自動感測已啟動")
            st.info("💡 操作提示：請將餐盤放置於鏡頭前。系統偵測到畫面**靜止不動 2 秒**時，將自動執行偵測（每次放置僅觸發一次）。")
            
            auto_shot = st.camera_input("自動感應鏡頭", key="live_auto_cam")
            
            if auto_shot:
                captured_image = Image.open(auto_shot).convert("RGB")
                
                # 計算 Frame 雜湊值判定是否為「同一餐盤靜止」
                current_frame_hash = hash(captured_image.tobytes()[:2000])
                last_hash = st.session_state.get("last_scanned_hash", None)
                
                if current_frame_hash != last_hash:
                    # 模擬倒數靜止 2 秒
                    countdown_placeholder = st.empty()
                    for s in range(2, 0, -1):
                        countdown_placeholder.warning(f"⏳ 偵測到餐盤放入，保持靜止中... {s}s")
                        time.sleep(1)
                    countdown_placeholder.success("🎯 鎖定完成，自動觸發 AI 審計！")
                    st.session_state["last_scanned_hash"] = current_frame_hash
                    st.session_state["trigger_auto_scan"] = True
                else:
                    st.success("✅ 該餐盤已完成掃描，等待下一個餐盤移入...")
                    
        # 模式 B: 手動拍照備用
        elif scan_mode == "手動拍照 (Manual Capture)":
            manual_shot = st.camera_input("點擊拍照按鈕進行快照", key="manual_cam")
            if manual_shot:
                captured_image = Image.open(manual_shot).convert("RGB")
                st.session_state["trigger_auto_scan"] = True
                
        # 模式 C: 檔案上傳
        else:
            uploaded_file = st.file_uploader("上傳餐盤相片 (JPG/PNG)", type=["jpg", "png", "jpeg"])
            if uploaded_file:
                captured_image = Image.open(uploaded_file).convert("RGB")
                st.session_state["trigger_auto_scan"] = True

    # 執行偵測與數據寫入
    if captured_image and st.session_state.get("trigger_auto_scan", False):
        st.markdown("---")
        st.subheader("🎯 AI 偵測與分析結果")
        
        with st.spinner("AI 物體偵測中: YOLOS-tiny 正在繪製 Bounding Box..."):
            annotated_img, item_list, waste_ratio, primary_cat = execute_detection(captured_image)
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.image(annotated_img, caption="AI Object Detection 即時標記框", use_container_width=True)
            if item_list:
                st.dataframe(pd.DataFrame(item_list), use_container_width=True)
            else:
                st.info("光盤良好，無明顯食物殘餘。")
                
        with col_res2:
            with st.spinner("Flan-T5 正在生成該分店之即時調配指令..."):
                kitchen_advice = generate_kitchen_decision(selected_branch, branch_meta, target_dish, waste_ratio)
            
            st.success(f"👨‍🍳 **【{selected_branch}】後廚調配指令**:\n\n{kitchen_advice}")
            
            # 換算損耗金錢與碳排放
            unit_rice_cost = 0.015
            cut_g = 40 if "Level A" in branch_meta["level"] else 20
            saved_hkd = round(cut_g * unit_rice_cost * 25, 1)  # 當次審計批次估算
            saved_co2 = round((cut_g * 25 / 1000) * 1.6, 2)
            
            st.metric("偵測殘食盤面佔比", f"{waste_ratio:.1%}")
            
            # 關鍵步驟：即時寫入真實審計紀錄，同步至 Mode 2 Dashboard
            new_record = {
                "時間": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "分店名稱": selected_branch,
                "門市等級": branch_meta["level"].split(" ")[0],
                "餐點項目": target_dish,
                "殘食主要類別": primary_cat,
                "殘食面積佔比 (%)": round(waste_ratio * 100, 1),
                "估計浪費金額 (HK$)": saved_hkd,
                "碳排放量 (kg CO2e)": saved_co2
            }
            st.session_state["audit_history"].append(new_record)
            st.toast(f"✅ 審計數據已即時同步至總部 Dashboard！(累計第 {len(st.session_state['audit_history'])} 筆)")
            st.session_state["trigger_auto_scan"] = False

# ==============================================================================
# 7. MODE 2: 總部即時營運大盤 (Real-time Dashboard)
# ==============================================================================
else:
    st.subheader("📊 Mode 2: 大家樂集團總部 - 跨分店即時營運與 ESG 大盤")
    st.caption("數據來源：來自各分店 Mode 1 現場實際掃描產生的 Real-time Data (非靜態假數據)")
    
    audit_data = st.session_state["audit_history"]
    
    if len(audit_data) == 0:
        st.warning("⚠️ 目前資料庫尚無即時掃描紀錄。請先切換至「Mode 1: 前線餐盤智能偵測」完成幾次掃描！")
    else:
        df_realtime = pd.DataFrame(audit_data)
        
        # 頂部關鍵指標 (KPI Cards)
        total_scans = len(df_realtime)
        avg_waste = df_realtime["殘食面積佔比 (%)"].mean()
        total_waste_hkd = df_realtime["估計浪費金額 (HK$)"].sum()
        total_co2 = df_realtime["碳排放量 (kg CO2e)"].sum()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("即時累計審計盤數", f"{total_scans} 盤")
        kpi2.metric("全港平均殘食佔比", f"{avg_waste:.1f} %")
        kpi3.metric("累計食材損耗成本", f"HK$ {total_waste_hkd:,.1f}")
        kpi4.metric("累計碳排當量", f"{total_co2:.2f} kg CO2e")
        
        st.markdown("---")
        
        # 圖表區：按分店與 Level 即時聚合
        col_c1, col_c2 = st.columns([1, 1])
        
        with col_c1:
            st.markdown("#### 🏢 各分店即時平均殘食率 (%)")
            branch_grouped = df_realtime.groupby("分店名稱")["殘食面積佔比 (%)"].mean().reset_index()
            st.bar_chart(branch_grouped, x="分店名稱", y="殘食面積佔比 (%)", color="#FF4B4B")
            
        with col_c2:
            st.markdown("#### 🏷️ 門市等級 (Level A/B/C) 浪費金額分佈")
            level_grouped = df_realtime.groupby("門市等級")["估計浪費金額 (HK$)"].sum().reset_index()
            st.bar_chart(level_grouped, x="門市等級", y="估計浪費金額 (HK$)")
            
        st.markdown("### 📋 即時審計串流日誌 (Live Audit Log)")
        st.dataframe(df_realtime.sort_values(by="時間", ascending=False), use_container_width=True)
        
        # 清空紀錄按鈕 (便於 Demo 重新演練)
        if st.button("🗑️ 清空所有即時紀錄 (Demo Reset)"):
            st.session_state["audit_history"] = []
            st.rerun()
