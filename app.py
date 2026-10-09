"""
ISOM5240 Deep Learning Business Applications with Python
Group 05: TrayZero+ | AI-Driven Food Waste Audit & Loyalty System
Target Company: Café de Coral Holdings Limited (大家樂集團)

Pipeline 1 (Vision): kktlau115/trayzero-swin-waste-regression
Pipeline 2 (NLP):    kktlau115/trayzero-flant5-sop-alert
"""

import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

# ==============================================================================
# 1. 頁面基本配置 (Page Configuration)
# ==============================================================================
st.set_page_config(
    page_title="TrayZero+ | 大家樂智能餐盤審計系統",
    page_icon="🍽️",
    layout="wide"
)

# ==============================================================================
# 2. 模型載入函數 (Load Chained Pipelines with Cache)
# ==============================================================================
@st.cache_resource
def load_pipelines():
    """
    載入已微調並發布於 Hugging Face 的兩大核心管線：
    Pipeline 1: 視覺餐盤殘食審計 (Swin Transformer)
    Pipeline 2: 智能廚房 SOP 與 Club 100 獎勵生成 (Flan-T5-Base)
    """
    device = 0 if torch.cuda.is_available() else -1
    
    # Pipeline 1: 影像分類 (5 分類殘食率)
    p1_model_id = "kktlau115/trayzero-swin-waste-regression"
    pipe1 = pipeline("image-classification", model=p1_model_id, device=device)
    
    # Pipeline 2: 文本生成 (智能 SOP 預警)
    p2_model_id = "kktlau115/trayzero-flant5-sop-alert"
    pipe2 = pipeline("text2text-generation", model=p2_model_id, device=device)
    
    return pipe1, pipe2

# ==============================================================================
# 3. 業務邏輯函數 (Defined Core Functions)
# ==============================================================================
def predict_waste_level(image: Image.Image, pipe1) -> tuple:
    """
    Pipeline 1 推論：輸入餐盤照片，輸出預測殘食等級與置信度
    """
    results = pipe1(image)
    top_pred = results[0]
    label = top_pred["label"]
    score = top_pred["score"]
    return label, score

def generate_sop_directive(dish_name: str, waste_label: str, branch: str, repeat_count: int, pipe2) -> str:
    """
    Pipeline 2 推論：輸入審計數據構造 Prompt，輸出智能廚房 SOP 與會員獎勵指令
    """
    prompt = (
        f"Generate kitchen SOP alert and customer incentive for Café de Coral: "
        f"Dish: {dish_name}, Waste: {waste_label}, "
        f"Repeated Occurrences: {repeat_count} in branch {branch}. Action:"
    )
    generated = pipe2(prompt, max_length=128)[0]["generated_text"]
    return generated

def calculate_club100_rewards(waste_label: str) -> dict:
    """
    根據 Pipeline 1 殘食判定結果，換算 Club 100 會員回饋與減碳效益
    """
    if "0_Zero_Waste" in waste_label:
        return {
            "tier": "極致光盤獎 (Ultra Clean)",
            "reward": "【$3 堂食現金券】+ 50 綠色積分",
            "co2_saved": "0.45 kg CO2e",
            "reverse_pos": "維持現有份量"
        }
    elif "1_Minor_Leftovers" in waste_label:
        return {
            "tier": "達標惜食獎 (Standard Clean)",
            "reward": "20 綠色積分",
            "co2_saved": "0.30 kg CO2e",
            "reverse_pos": "維持現有份量"
        }
    elif "2_Partial_Leftover" in waste_label:
        return {
            "tier": "環保同行 (Moderate)",
            "reward": "10 綠色積分",
            "co2_saved": "0.15 kg CO2e",
            "reverse_pos": "下次點餐推薦「少飯」"
        }
    elif "3_Half_Eaten" in waste_label:
        return {
            "tier": "份量調整提示 (High Leftover)",
            "reward": "5 綠色積分",
            "co2_saved": "0.05 kg CO2e",
            "reverse_pos": "已於自助點餐機觸發「少飯扣減 $2」"
        }
    else:
        return {
            "tier": "品質反饋 (Heavy Waste)",
            "reward": "發送產品滿意度調查問卷",
            "co2_saved": "0.00 kg CO2e",
            "reverse_pos": "通知後廚主管核查該批次品質"
        }

# ==============================================================================
# 4. 主程式介面進入點 (Main Function)
# ==============================================================================
def main():
    st.title("🍽️ TrayZero+ | 大家樂智能餐盤廚餘審計與會員獎勵系統")
    st.caption("HKUST ISOM5240 Group 05 | 香港科技大學 深度學習商業應用 成果展示")
    st.write("---")

    # 側邊欄：門市與餐點模擬設定
    st.sidebar.header("🏪 大家樂門市環境設定")
    selected_branch = st.sidebar.selectbox(
        "選擇營業分店 (Branch)",
        ["沙田新城市廣場店 (Sha Tin)", "中環威靈頓街店 (Central CBD)", "香港科技大學店 (HKUST)", "將軍澳 Popcorn 店 (TKO)"]
    )
    selected_dish = st.sidebar.selectbox(
        "選擇審計菜式 (Target Dish)",
        [
            "Baked Pork Chop Rice (一哥焗豬扒飯)",
            "Curry Beef Brisket Rice (咖喱牛腩飯)",
            "Scrambled Egg Shrimp Rice (滑蛋蝦仁飯)",
            "Spaghetti Bolognese (焗肉醬意粉)",
            "Minced Pork Patty Rice (香辣肉燥肉餅飯)",
            "Kart Noodle (車仔麵)"
        ]
    )
    repeat_count = st.sidebar.slider("今日該菜式殘食重複通報次數", min_value=1, max_value=5, value=2)

    # 載入模型
    with st.spinner("🚀 正在連線 Hugging Face 載入微調雙管線模型..."):
        pipe1, pipe2 = load_pipelines()

    # 主要工作區：兩欄佈局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📷 步驟一：拍攝或上傳餐盤照片")
        uploaded_file = st.file_uploader("上傳餐盤照片 (.jpg / .png)", type=["jpg", "jpeg", "png"])
        
        # 預設展示照片 (若用戶未上傳)
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            # 建立一張模擬餐盤佔位圖，保證直接 Run 也能展示
            image = Image.new("RGB", (224, 224), color=(235, 230, 220))
            st.info("💡 提示：可直接上傳餐盤照片，或使用系統預設測試圖片進行推論。")
            
        st.image(image, caption="餐盤即時影像", use_container_width=True)

    with col2:
        st.subheader("🤖 步驟二：執行雙管線端到端推論")
        if st.button("開始智能分析 (Run Audit Pipeline)", type="primary"):
            # 1. 執行 Pipeline 1
            with st.spinner("執行 Pipeline 1 (Swin-Tiny 視覺審計)..."):
                waste_label, confidence = predict_waste_level(image, pipe1)
                rewards_data = calculate_club100_rewards(waste_label)
            
            # 2. 執行 Pipeline 2
            with st.spinner("執行 Pipeline 2 (Flan-T5 智能 SOP 決策)..."):
                sop_output = generate_sop_directive(selected_dish, waste_label, selected_branch, repeat_count, pipe2)

            # 展示 Pipeline 1 結果
            st.success("✅ Pipeline 1 視覺審計完成！")
            st.metric("預測殘食等級", waste_label, f"置信度: {confidence:.2%}")

            # 展示 Club 100 獎勵
            st.info(f"🎁 **Club 100 會員回饋**：{rewards_data['tier']}\n\n"
                    f"* **獲得獎勵**：{rewards_data['reward']}\n"
                    f"* **ESG 減碳效益**：已節省約 {rewards_data['co2_saved']}\n"
                    f"* **點餐機逆向優惠 (Reverse-POS)**：{rewards_data['reverse_pos']}")

            # 展示 Pipeline 2 結果
            st.markdown("---")
            st.subheader("📋 Pipeline 2 智能生成後廚 SOP 指令")
            st.warning(f"**後廚指令**：\n\n{sop_output}")

    st.write("---")
    st.caption("模型託管於 Hugging Face: [Pipeline 1](https://huggingface.co/kktlau115/trayzero-swin-waste-regression) | [Pipeline 2](https://huggingface.co/kktlau115/trayzero-flant5-sop-alert)")

# ==============================================================================
# 5. 程式進入點
# ==============================================================================
if __name__ == "__main__":
    main()
