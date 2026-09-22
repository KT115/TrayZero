def generate_actionable_recommendations(
    branch_name: str, branch_level: str, base_rice_g: float, 
    strategy: str, dish_name: str, waste_ratio: float, 
    primary_category: str, engine: dict
) -> dict:
    """
    生成具體可執行的下一步應對清單 (Actionable Next Steps)
    包含：後廚 SOP、前台 POS、經理備料決策
    """
    # 1. 建立精確的業務量化應對邏輯 (Rule-based Core)
    actions = {
        "kitchen_sop": "",
        "pos_intervention": "",
        "manager_prep": "",
        "priority_level": "正常 (Normal)"
    }
    
    if "白飯" in primary_category or waste_ratio >= 0.35:
        actions["priority_level"] = "🔴 高度警報 (High Priority)"
        cut_grams = 40 if "Level A" in str(branch_level) else 25
        target_grams = int(base_rice_g - cut_grams)
        
        actions["kitchen_sop"] = (
            f"【更換打飯工具】即刻停用原裝飯勺，改用特定規格量勺，"
            f"將標準打飯量由 {base_rice_g}g 下調至 {target_grams}g（單份減量 {cut_grams}g）。"
        )
        actions["pos_intervention"] = (
            "【點餐機即時促銷】自助點餐機（Kiosk）即刻彈窗提示「少飯減扣 HK$ 2」或「可換配無糖熱飲」，"
            "將小食量顧客主動引流至輕量裝，從源頭降低出餐量。"
        )
        actions["manager_prep"] = (
            f"【晚市/明日備料調整】通知煮飯崗位，將下一輪電飯煲蒸煮量調減 2 鍋（約下調 12% 產能），"
            f"預計單日可直接節省大米耗損約 HK$ {round(cut_grams * 0.015 * 800, 0):,.0f}。"
        )

    elif "主菜" in primary_category or "肉類" in primary_category:
        actions["priority_level"] = "🟠 品質警報 (Quality Alert)"
        actions["kitchen_sop"] = (
            f"【烹調品管即時複核】主菜肉排剩餘率異常！主廚須立刻抽驗炸焗爐溫度（標準核心溫度需達 75°C），"
            "檢查是否肉質過柴或醬汁淋灑不足；要求下一批次焗烤時間縮短 45 秒。"
        )
        actions["pos_intervention"] = (
            "【菜品滿意度回饋】前台收銀員於顧客取餐時留意反饋；若為醬汁偏鹹，廚房立即稀釋下一桶調味汁。"
        )
        actions["manager_prep"] = (
            "【供應鏈批次登記】在後台系統記錄當前肉排的進貨批號（Batch ID），"
            "若晚市連續兩輪檢出肉類殘留超標，即時向大家樂大埔中央廚房發出品質異動通報。"
        )

    elif "配菜" in primary_category or "醬汁" in primary_category:
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

    else: # 光盤或殘留極低
        actions["priority_level"] = "🟢 營運標準 (Optimal)"
        actions["kitchen_sop"] = (
            f"【維持出餐標竿】目前出餐規格 ({base_rice_g}g) 與口感表現完美，顧客吃完率極高，嚴禁廚房隨意減量。"
        )
        actions["pos_intervention"] = (
            "【主打推薦】該餐點維持點餐機首頁熱銷推薦輪播。"
        )
        actions["manager_prep"] = (
            "【確保庫存充足】維持標準進貨與解凍備料量，防止尖峰時段提前斷貨。"
        )

    # 2. 結合 Flan-T5 語言模型生成更靈活的總結建議
    prompt = (
        f"You are the senior operations director of Cafe de Coral. "
        f"Store: {branch_name}, Dish: {dish_name}, Primary Waste: {primary_category}, Ratio: {waste_ratio:.1%}. "
        f"Provide a brief executive memo on what immediate step the store manager should execute now."
    )
    inputs = engine["tokenizer"](prompt, return_tensors="pt", max_length=512, truncation=True).to(engine["device"])
    with torch.no_grad():
        outputs = engine["generator"].generate(**inputs, max_new_tokens=70, do_sample=False)
    actions["executive_memo"] = engine["tokenizer"].decode(outputs[0], skip_special_tokens=True)

    return actions
