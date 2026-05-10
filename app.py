# ============================================================
# app.py — Streamlit PPE Detection Web App
# Run with: streamlit run app.py
#
# SETUP on local machine:
#   pip install streamlit ultralytics opencv-python Pillow
#   streamlit run app.py
#
# SETUP on Colab (for demo):
#   !pip install streamlit pyngrok ultralytics opencv-python -q
#   !ngrok authtoken YOUR_NGROK_TOKEN
#   Then run the ngrok tunnel cell below
# ============================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# ---- Page Config ----
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #FF6B35;
    }
    .violation-box {
        background: #ffe0e0;
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #cc0000;
    }
    .compliant-box {
        background: #e0ffe0;
        border: 2px solid #44aa44;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #006600;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background-color: #e55a25;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL — cached so it loads only once
# ============================================================

@st.cache_resource
def load_model(model_path):
    """Load YOLO model once and cache it."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ============================================================
# PPE COMPLIANCE LOGIC
# ============================================================

def check_ppe_compliance(detections, class_names):
    """
    Determine if each detected person is wearing a helmet.
    Returns: (violation_count, person_count, compliance_details)
    """
    persons = []
    helmets = []
    vests = []

    if detections.boxes is None or len(detections.boxes) == 0:
        return 0, 0, []

    for box in detections.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        cls_name = class_names[cls_id].lower()
        coords = box.xyxy[0].cpu().numpy()

        if 'person' in cls_name:
            persons.append({'box': coords, 'conf': conf})
        elif 'helmet' in cls_name:
            helmets.append({'box': coords, 'conf': conf})
        elif 'vest' in cls_name or 'safety' in cls_name:
            vests.append({'box': coords, 'conf': conf})

    compliance_details = []
    violation_count = 0

    for i, person in enumerate(persons):
        px1, py1, px2, py2 = person['box']
        head_zone_y = py1 + (py2 - py1) * 0.4  # top 40% of person = head region

        # Check helmet
        has_helmet = False
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet['box']
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            if px1 <= hcx <= px2 and py1 <= hcy <= head_zone_y:
                has_helmet = True
                break

        # Check vest (vest center should be in lower 60% of person)
        has_vest = False
        vest_zone_y = py1 + (py2 - py1) * 0.4
        for vest in vests:
            vx1, vy1, vx2, vy2 = vest['box']
            vcx = (vx1 + vx2) / 2
            vcy = (vy1 + vy2) / 2
            if px1 <= vcx <= px2 and vest_zone_y <= vcy <= py2:
                has_vest = True
                break

        is_compliant = has_helmet  # primary check: helmet
        if not is_compliant:
            violation_count += 1

        compliance_details.append({
            'person_id': i + 1,
            'has_helmet': has_helmet,
            'has_vest': has_vest,
            'is_compliant': is_compliant,
            'box': person['box']
        })

    return violation_count, len(persons), compliance_details


def draw_annotated_image(image_array, result, compliance_details, class_names):
    """Draw YOLO boxes + compliance overlay on image."""
    # Get YOLO's automatic annotation
    annotated = result.plot()  # BGR
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Draw compliance indicators per person
    for detail in compliance_details:
        x1, y1, x2, y2 = map(int, detail['box'])
        color = (0, 180, 0) if detail['is_compliant'] else (255, 30, 30)

        # Thick border around each person
        cv2.rectangle(annotated_rgb, (x1, y1), (x2, y2), color, 3)

        # Label
        label = f"P{detail['person_id']}: {'✓ Safe' if detail['is_compliant'] else '✗ Violation'}"
        cv2.putText(annotated_rgb, label, (x1, max(y1 - 8, 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return annotated_rgb


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<div class="main-header">🦺 PPE Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Construction Site Safety Monitoring | YOLOv8n | Real-time PPE Compliance</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/128/hard-hat.png", width=80)
        st.title("⚙️ Settings")

        st.subheader("Model")
        model_path = st.text_input(
            "Model weights path",
            value="ppe_best.pt",
            help="Path to your trained YOLOv8 .pt file"
        )

        st.subheader("Detection Parameters")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=0.9,
            value=0.35, step=0.05,
            help="Lower = more detections (more false positives). Higher = fewer, surer detections."
        )
        iou_threshold = st.slider(
            "IoU (NMS) Threshold",
            min_value=0.1, max_value=0.9,
            value=0.45, step=0.05,
            help="Controls overlap between boxes. Lower = fewer overlapping boxes."
        )

        st.subheader("About")
        st.info("""
        **Project:** PPE Detection System  
        **Model:** YOLOv8n  
        **Dataset:** SH17 (4700 images)  
        **Classes:** Helmet, Safety Vest, Person, Head  
        **Developer:** Your Name  
        **Internship Project:** 2024
        """)

        st.subheader("📊 Model Info")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | mAP@0.5 | ~55% |
        | FPS (T4) | ~80 |
        | Params | 3.2M |
        | Size | ~6MB |
        """)

    # Load model
    model = load_model(model_path)
    if model is None:
        st.error("⚠️ Could not load model. Please check the path in the sidebar.")
        st.code("# Download model weights first:\n# Place 'ppe_best.pt' in same folder as app.py")
        st.stop()

    st.success(f"✅ Model loaded: `{model_path}`")

    # ---- Mode Selection ----
    mode = st.radio(
        "Select Input Mode",
        ["📷 Upload Image", "🎬 Upload Video", "📸 Webcam (Demo)"],
        horizontal=True
    )

    # ================================================================
    # MODE 1: IMAGE UPLOAD
    # ================================================================
    if mode == "📷 Upload Image":
        st.subheader("📷 Image PPE Detection")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Upload a construction site image",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
            )

            if uploaded_file:
                # Load image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Original Image", use_column_width=True)

        if uploaded_file and st.button("🔍 Detect PPE", type="primary"):
            with col2:
                with st.spinner("Running PPE detection..."):
                    # Convert PIL to numpy array
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    # Inference
                    t_start = time.time()
                    results = model.predict(
                        img_bgr,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=640,
                        verbose=False
                    )
                    inference_time = (time.time() - t_start) * 1000

                    result = results[0]

                    # Compliance check
                    violation_count, person_count, details = check_ppe_compliance(result, model.names)

                    # Draw annotations
                    annotated = draw_annotated_image(img_array, result, details, model.names)
                    st.image(annotated, caption="Detection Result", use_column_width=True)

                # Metrics row
                st.markdown("---")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("👷 People Detected", person_count)
                mc2.metric("⚠️ Violations", violation_count,
                          delta=f"-{violation_count}" if violation_count > 0 else None,
                          delta_color="inverse")
                mc3.metric("✅ Compliant", person_count - violation_count)
                mc4.metric("⏱️ Inference", f"{inference_time:.0f}ms")

                # Compliance status
                if violation_count == 0 and person_count > 0:
                    st.markdown('<div class="compliant-box">✅ PPE COMPLIANT — All workers are wearing required PPE</div>',
                               unsafe_allow_html=True)
                elif violation_count > 0:
                    st.markdown(f'<div class="violation-box">❌ PPE VIOLATION DETECTED — {violation_count} of {person_count} worker(s) at risk</div>',
                               unsafe_allow_html=True)
                elif person_count == 0:
                    st.warning("⚠️ No persons detected in the image. Try lowering the confidence threshold.")

                # Detailed table
                if details:
                    st.subheader("📋 Per-Worker Analysis")
                    rows = []
                    for d in details:
                        rows.append({
                            "Worker ID": f"Worker {d['person_id']}",
                            "Helmet": "✅" if d['has_helmet'] else "❌",
                            "Safety Vest": "✅" if d['has_vest'] else "❓",
                            "Status": "✅ COMPLIANT" if d['is_compliant'] else "❌ VIOLATION"
                        })
                    st.table(rows)

                # Download annotated image
                from PIL import Image as PILImage
                import io
                annotated_pil = PILImage.fromarray(annotated)
                buf = io.BytesIO()
                annotated_pil.save(buf, format='JPEG', quality=95)
                st.download_button(
                    "⬇️ Download Result",
                    data=buf.getvalue(),
                    file_name="ppe_detection_result.jpg",
                    mime="image/jpeg"
                )

    # ================================================================
    # MODE 2: VIDEO UPLOAD
    # ================================================================
    elif mode == "🎬 Upload Video":
        st.subheader("🎬 Video PPE Detection")
        st.info("⚠️ Video processing is slow on CPU. Best run on Google Colab with GPU. Here we process a preview (first 100 frames).")

        uploaded_video = st.file_uploader(
            "Upload a construction site video",
            type=['mp4', 'avi', 'mov', 'mkv']
        )

        max_frames = st.slider("Max frames to process", 50, 300, 100, 10)

        if uploaded_video and st.button("🎬 Process Video", type="primary"):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_input = tfile.name

            output_path = temp_input.replace('.mp4', '_result.mp4')

            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_placeholder = st.empty()

            cap = cv2.VideoCapture(temp_input)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            total_violations = 0
            total_persons = 0

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, conf=conf_threshold, iou=iou_threshold,
                                       imgsz=640, verbose=False)
                result = results[0]
                violations, persons, details = check_ppe_compliance(result, model.names)
                total_violations += violations
                total_persons += persons

                annotated = result.plot()

                # Overlay
                color = (0, 200, 0) if violations == 0 else (0, 0, 255)
                cv2.rectangle(annotated, (0, 0), (width, 45), color, -1)
                status = 'PPE COMPLIANT' if violations == 0 else f'VIOLATION: {violations} at risk'
                cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(annotated, f'Frame {frame_count+1}', (width-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                out.write(annotated)
                frame_count += 1
                progress_bar.progress(frame_count / total)
                status_text.text(f"Processing frame {frame_count}/{total}...")

            cap.release()
            out.release()

            stats_placeholder.success(f"""
            ✅ Video processed!
            - **Frames analyzed:** {frame_count}
            - **Total violations detected:** {total_violations}
            - **Average violation rate:** {total_violations/max(frame_count,1)*100:.1f}% of frames
            """)

            # Show download
            with open(output_path, 'rb') as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f.read(),
                    file_name="ppe_detection_video.mp4",
                    mime="video/mp4"
                )

            os.unlink(temp_input)
            if os.path.exists(output_path):
                os.unlink(output_path)

    # ================================================================
    # MODE 3: WEBCAM (Demo / Local only)
    # ================================================================
    elif mode == "📸 Webcam (Demo)":
        st.subheader("📸 Real-time Webcam Detection")
        st.warning("⚠️ Webcam mode works only when running locally (`streamlit run app.py`). Not supported in cloud deployments.")
        st.info("""
        To run locally:
        ```bash
        pip install streamlit ultralytics opencv-python Pillow
        streamlit run app.py
        ```
        Then select Webcam mode and click Start.
        """)

        run = st.checkbox("▶️ Start Webcam")
        frame_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam. Check permissions.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=conf_threshold, iou=iou_threshold,
                                           imgsz=640, verbose=False, stream=False)
                    result = results[0]
                    violations, persons, details = check_ppe_compliance(result, model.names)
                    annotated = result.plot()

                    color = (0, 200, 0) if violations == 0 else (0, 0, 255)
                    cv2.rectangle(annotated, (0, 0), (640, 40), color, -1)
                    status = '✓ COMPLIANT' if violations == 0 else f'✗ VIOLATION: {violations} at risk'
                    cv2.putText(annotated, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                           channels="RGB", use_column_width=True)
                cap.release()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
    🦺 PPE Detection System | Built with YOLOv8 + Streamlit | Solo Internship Project<br>
    Dataset: SH17 PPE Detection | Model: YOLOv8n (3.2M params) | Classes: Helmet, Safety Vest, Person, Head
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
