import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置页面配置 - 宽模式
st.set_page_config(layout="wide")

# 自定义 CSS - 背景和样式
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4efe9 100%);
        min-height: 100vh;
    }
    .stHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        background: rgba(255,255,255,0.8);
    }
    .stSuccess {
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 健康数据
health_dict = {
    "chicken_curry": {"calories": 120, "protein": "8g", "fat": "6g", "level": "★★★☆☆", "advice": "咖喱较咸，适量食用。"},
    "chicken_wings": {"calories": 290, "protein": "19g", "fat": "22g", "level": "★★☆☆☆", "advice": "油炸食品，建议减少摄入。"},
    "fried_rice": {"calories": 180, "protein": "5g", "fat": "7g", "level": "★★☆☆☆", "advice": "米饭升糖较快，注意分量。"},
    "grilled_salmon": {"calories": 200, "protein": "22g", "fat": "12g", "level": "★★★★☆", "advice": "富含Omega-3，推荐食用！"},
    "hamburger": {"calories": 250, "protein": "13g", "fat": "12g", "level": "★★☆☆☆", "advice": "高热量食物，适量食用。"},
    "ice_cream": {"calories": 210, "protein": "4g", "fat": "11g", "level": "★★☆☆☆", "advice": "含糖量较高，不宜过量。"},
    "pizza": {"calories": 266, "protein": "11g", "fat": "10g", "level": "★★★☆☆", "advice": "均衡饮食，适量食用。"},
    "ramen": {"calories": 450, "protein": "10g", "fat": "20g", "level": "★★☆☆☆", "advice": "热量较高，注意控制分量。"},
    "steak": {"calories": 270, "protein": "26g", "fat": "19g", "level": "★★★☆☆", "advice": "优质蛋白来源，搭配蔬菜更佳。"},
    "sushi": {"calories": 150, "protein": "5g", "fat": "1g", "level": "★★★★☆", "advice": "低脂肪，推荐食用！"}
}

CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi']

# 加载单个模型
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model_label_smoothing.keras')
    return model

model = load_model()

# 页面标题
st.markdown("""
    <div class="stHeader">
        <h1 style="color: white; text-align: center; margin: 0;">🍔 食物图像识别与健康评估系统</h1>
    </div>
    """, unsafe_allow_html=True)

# 创建两列布局 - 左侧上传区域，右侧结果区域
col_main, _ = st.columns([5, 1])  # 主要内容占80%

with col_main:
    # 上传图片
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)

        # 创建两列显示原图和预测结果
        col_img, col_result = st.columns([1, 2])
        
        with col_img:
            st.image(image, caption="原图", width=250)
        
        with col_result:
            # 预处理
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
            
            # 预测
            pred = model.predict(img_array, verbose=0)[0]
            pred_index = np.argmax(pred)
            pred_class = CLASS_NAMES[pred_index]
            confidence = np.max(pred)
            
            # 分类结果
            st.markdown("## 🎯 分类结果")
            st.markdown(f"<span class='stSuccess'>{pred_class.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
            st.write(f"**置信度**：{confidence * 100:.2f}%")

        # Top-5 概率条形图
        st.markdown("---")
        st.markdown("## 📊 Top-5 预测概率")
        top5_indices = np.argsort(pred)[::-1][:5]
        top5_probs = pred[top5_indices]
        top5_names = [CLASS_NAMES[i].replace('_', ' ').title() for i in top5_indices]
        
        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.barh(top5_names, top5_probs * 100, color=['#FF6B6B' if i == pred_index else '#4ECDC4' for i in top5_indices])
        ax.set_xlabel('Probability (%)', fontsize=12)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='both', labelsize=12)
        
        for bar, prob in zip(bars, top5_probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', va='center', fontsize=11)
        
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        # 健康评估仪表盘
        st.markdown("---")
        st.markdown("## 🥗 健康评估")
        if pred_class in health_dict:
            info = health_dict[pred_class]
            
            # 创建三列布局
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔥 热量")
                calories = info["calories"]
                calories_percent = min(calories / 500 * 100, 100)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                wedge = patches.Wedge((0.5, 0.5), 0.4, 90, 90 - calories_percent * 3.6, width=0.15, color='#FF6B6B')
                ax.add_patch(wedge)
                ax.text(0.5, 0.5, f'{calories}\nkcal', ha='center', va='center', fontsize=14)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 💪 蛋白质")
                protein_g = int(info["protein"].replace('g', ''))
                protein_percent = min(protein_g / 30 * 100, 100)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                wedge = patches.Wedge((0.5, 0.5), 0.4, 90, 90 - protein_percent * 3.6, width=0.15, color='#4ECDC4')
                ax.add_patch(wedge)
                ax.text(0.5, 0.5, f'{protein_g}\ng', ha='center', va='center', fontsize=14)
                st.pyplot(fig)
            
            with col3:
                st.markdown("### 🧈 脂肪")
                fat_g = int(info["fat"].replace('g', ''))
                fat_percent = min(fat_g / 30 * 100, 100)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                wedge = patches.Wedge((0.5, 0.5), 0.4, 90, 90 - fat_percent * 3.6, width=0.15, color='#FFE66D')
                ax.add_patch(wedge)
                ax.text(0.5, 0.5, f'{fat_g}\ng', ha='center', va='center', fontsize=14)
                st.pyplot(fig)
            
            # 健康等级和建议
            st.markdown("---")
            st.markdown("### ⭐ 健康等级")
            st.markdown(f"<h1 style='text-align: center;'>{info['level']}</h1>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                    <p style='color: white; font-size: 1.1rem; text-align: center; margin: 0;'>💡 {info['advice']}</p>
                </div>
                """, unsafe_allow_html=True)
