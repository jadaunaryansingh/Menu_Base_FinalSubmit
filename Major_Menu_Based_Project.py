# flask_streamlit_menu_app.py
import streamlit as st
from flask import Flask, jsonify
import threading
import time
import requests
import os
import shutil
import cv2
import stat
import numpy as np
from email.mime.text import MIMEText
import smtplib
from googlesearch import search
import pywhatkit as pw
from twilio.rest import Client
import stat
import time

try:
    import pwd
    import grp
except ImportError:
    pwd = None
    grp = None
# ------------------ Flask API with Decorators ------------------
app = Flask(__name__)
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
@app.route("/api/ping")
@log_decorator
def ping():
    return jsonify({"message": "pong"})
# ------------------ Streamlit Interface ------------------
st.set_page_config(page_title="Mega Menu App", layout="wide")
st.title("🚀 Multi-Tool Mega App (Flask + Streamlit)")
menu = st.sidebar.selectbox("Choose Feature", [
    "📞 Twilio Call", "💬 Send SMS", "🟢 WhatsApp Message",
    "📧 Send Email (pywhatkit)","🐳 Remote Docker Command Center","🎨 Draw Grid Image", "🔄 Face Swap",
    "🌍 Download Website HTML", "🔗 Post on LinkedIn","🤖 Machine Learning Tasks", "🔎 Google Search","🕸️</>🛠️JS Tasks", "🗂 File Manager", "📡 Ping API","🔐 Linux Command Executor","🧠 Remote File Control (Cross OS)"
])

if menu == "🕸️</>🛠️JS Tasks":
    import streamlit as st
    import streamlit.components.v1 as components
    import base64
    import json

    st.set_page_config(page_title="JS + Docker Task Hub", layout="wide")
    st.title("🌐 JavaScript & Browser Tasks Hub")

    tasks = [
    "Take Photo Using JavaScript",
    "Send Email Using JS (via mailto link)",
    "Send Captured Photo via Email (mailto with attachment not feasible, demo only)",
    "Record Video on Button Click and Send via Email (mailto with link demo)",
    "Send WhatsApp Message Using JavaScript",
    "Send SMS Using JavaScript (via sms: URL)",
    "Show Current Jio (Geo) Location",
    "Show Location on Google Maps (Live View)",
    "Show Route from Current Location to Destination",
    "Show Nearby Grocery Stores on Google Maps",
    "Fetch Last Email Info from Gmail (Mock Demo)",
    "Post from Chrome Browser to Social Media (Demo)",
    "Track Most Viewed Products and Show 'Recommended'",
    "Track Skipped Products and Use View Time for Reports (Demo)",
    "Get IP Address and Location on Click"
    ]

    task = st.selectbox("Select a JS Task to Run", tasks)

# Helper: Embed JS in Streamlit with communication via window.parent.postMessage
    def js_component(js_code, height=400):
        html_code = f"""
        <html><body>
        <script>
        {js_code}
        </script>
        </body></html>
        """
        components.html(html_code, height=height)

# 1. Take Photo Using JavaScript
    if task == "Take Photo Using JavaScript":
        st.write("📸 Click below to activate your webcam and capture a photo.")
        js_code = """
        async function init() {
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            document.body.innerHTML = '<button id="startBtn">Start Camera</button><button id="snapBtn" disabled>Capture Photo</button><br><img id="photo" style="max-width: 300px;"/>';

            const startBtn = document.getElementById('startBtn');
            const snapBtn = document.getElementById('snapBtn');
            const photo = document.getElementById('photo');

            startBtn.onclick = async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();
                document.body.appendChild(video);
                snapBtn.disabled = false;
            };

            snapBtn.onclick = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                const dataURL = canvas.toDataURL('image/png');
                photo.src = dataURL;
                window.parent.postMessage({func:'photoCaptured', data: dataURL}, '*');
            };
        }
        init();
        """
        js_component(js_code, 450)

    # Receive photo base64 from JS if needed - (Streamlit can't directly receive from JS like this, so demo only)

# 2. Send Email Using JS (via mailto)
    elif task == "Send Email Using JS (via mailto link)":
        st.write("Send an email using your default email client:")
        to = st.text_input("To Email", "example@example.com")
        subject = st.text_input("Subject", "Hello from Streamlit")
        body = st.text_area("Body", "This is a test email.")
        if st.button("Open Email Client"):
            mailto = f"mailto:{to}?subject={subject}&body={body}"
            st.markdown(f"[Click here to open mail client]({mailto})")

# 3. Send Captured Photo via Email (mailto with attachment not possible)
    elif task == "Send Captured Photo via Email (mailto with attachment not feasible, demo only)":
        st.write("Due to browser limitations, sending attachments via mailto is not supported.")
        st.info("Use 'Take Photo' task, save the photo manually, and attach it in your email client.")
        st.write("This task cannot be fully automated with mailto links.")

# 4. Record Video on Button Click and Send via Email (demo)
    elif task == "Record Video on Button Click and Send via Email (mailto with link demo)":
        st.write("Record a short video and get a mailto link to send an email (video upload not automated).")

        js_code = """
        let mediaRecorder;
        let recordedChunks = [];

        document.body.innerHTML = `
        <video id="preview" width="320" height="240" autoplay muted></video><br/>
        <button id="start">Start Recording</button>
        <button id="stop" disabled>Stop Recording</button>
        <video id="playback" width="320" height="240" controls style="display:none"></video>
        `;

        const preview = document.getElementById('preview');
        const playback = document.getElementById('playback');
        const start = document.getElementById('start');
        const stop = document.getElementById('stop');

        async function init() {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        preview.srcObject = stream;

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = e => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            playback.src = url;
            playback.style.display = 'block';
            recordedChunks = [];
        };

        start.onclick = () => {
            mediaRecorder.start();
            start.disabled = true;
            stop.disabled = false;
        };

        stop.onclick = () => {
            mediaRecorder.stop();
            start.disabled = false;
            stop.disabled = true;
        };
        }
        init();
        """
        js_component(js_code, 480)
        st.info("After recording, download the video and attach it manually to your email client.")

# 5. Send WhatsApp Message Using JavaScript
    elif task == "Send WhatsApp Message Using JavaScript":
        st.write("Send WhatsApp message via WhatsApp Web URL scheme:")
        number = st.text_input("Enter phone number (with country code, e.g. 919876543210)", "")
        message = st.text_area("Message", "Hello from Streamlit!")
        if st.button("Open WhatsApp Web"):
            if number.strip() == "":
                st.error("Please enter a valid phone number")
            else:
                wa_url = f"https://wa.me/{number}?text={message.replace(' ', '%20')}"
                st.markdown(f"[Open WhatsApp Chat]({wa_url})", unsafe_allow_html=True)

# 6. Send SMS Using JavaScript (sms: URL)
    elif task == "Send SMS Using JavaScript (via sms: URL)":
        st.write("Send SMS using default SMS client:")
        number = st.text_input("Enter phone number", "")
        message = st.text_area("Message", "Hello from Streamlit!")
        if st.button("Open SMS App"):
            if number.strip() == "":
                st.error("Please enter a valid phone number")
            else:
                sms_url = f"sms:{number}?body={message.replace(' ', '%20')}"
                st.markdown(f"[Open SMS App]({sms_url})", unsafe_allow_html=True)

# 7. Show Current Jio (Geo) Location
    elif task == "Show Current Jio (Geo) Location":
        js_code = """
        navigator.geolocation.getCurrentPosition(function(pos) {
        const coords = pos.coords;
        document.body.innerHTML = `<h3>Your Location:</h3>
        <p>Latitude: ${coords.latitude}</p>
        <p>Longitude: ${coords.longitude}</p>
        <p>Accuracy: ${coords.accuracy} meters</p>`;
        }, function(err) {
        document.body.innerHTML = '<p>Geolocation error: ' + err.message + '</p>';
        });
        """
        js_component(js_code, 200)

# 8. Show Location on Google Maps (Live View)
    elif task == "Show Location on Google Maps (Live View)":
        js_code = """
        navigator.geolocation.watchPosition(function(pos) {
        const coords = pos.coords;
        const lat = coords.latitude;
        const lng = coords.longitude;
        document.body.innerHTML = `<iframe width="600" height="450"
            src="https://maps.google.com/maps?q=${lat},${lng}&hl=en&z=15&output=embed"></iframe>`;
        }, function(err) {
        document.body.innerHTML = '<p>Geolocation error: ' + err.message + '</p>';
        });
        """
        js_component(js_code, 500)

# 9. Show Route from Current Location to Destination
    elif task == "Show Route from Current Location to Destination":
        dest = st.text_input("Enter destination address (e.g., 'India Gate, Delhi')", "")
        if dest.strip():
            js_code = f"""
            navigator.geolocation.getCurrentPosition(function(pos) {{
            const lat = pos.coords.latitude;
            const lng = pos.coords.longitude;
            const destination = encodeURIComponent("{dest}");
            document.body.innerHTML = `<iframe width="600" height="450"
                src="https://www.google.com/maps/embed/v1/directions?key=YOUR_GOOGLE_MAPS_API_KEY&origin=${{lat}},${{lng}}&destination=${{destination}}&mode=driving">
                </iframe>`;
            }}, function(err) {{
            document.body.innerHTML = '<p>Geolocation error: ' + err.message + '</p>';
            }});
            """
            st.warning("**Replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual Google Maps API key in the code.**")
            js_component(js_code, 500)
        else:
            st.info("Enter destination address to show route.")

# 10. Show Nearby Grocery Stores on Google Maps
    elif task == "Show Nearby Grocery Stores on Google Maps":
        js_code = """
        navigator.geolocation.getCurrentPosition(function(pos) {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        const map_url = `https://www.google.com/maps/search/grocery+stores/@${lat},${lng},15z`;
        document.body.innerHTML = `<a href="${map_url}" target="_blank">Open Nearby Grocery Stores in Google Maps</a>`;
        }, function(err) {
        document.body.innerHTML = '<p>Geolocation error: ' + err.message + '</p>';
        });
        """
        js_component(js_code, 200)

# 11. Fetch Last Email Info from Gmail (Mock Demo)
    elif task == "Fetch Last Email Info from Gmail (Mock Demo)":
        st.write("Due to OAuth and API restrictions, this is a mock demo.")
        st.write("Normally requires Google OAuth and Gmail API setup.")
        st.json({
            "from": "example@gmail.com",
            "subject": "Welcome to Gmail API Demo",
            "snippet": "This is a mock email snippet from the latest email."
        })

# 12. Post from Chrome Browser to Social Media (Demo)
    elif task == "Post from Chrome Browser to Social Media (Demo)":
        social = st.selectbox("Select Social Media", ["Twitter", "Instagram", "Facebook"])
        post_text = st.text_area("Post Text", "Hello from Streamlit!")
        if st.button("Open Social Media Post Page"):
            url = ""
            if social == "Twitter":
                url = f"https://twitter.com/intent/tweet?text={post_text.replace(' ', '%20')}"
            elif social == "Facebook":
                url = f"https://www.facebook.com/sharer/sharer.php?u=&quote={post_text.replace(' ', '%20')}"
            elif social == "Instagram":
                st.warning("Instagram does not support direct web posting via URL. Use mobile app.")
            if url:
                st.markdown(f"[Open {social} Post Page]({url})", unsafe_allow_html=True)

# 13. Track Most Viewed Products and Show 'Recommended'
    elif task == "Track Most Viewed Products and Show 'Recommended'":
        if 'views' not in st.session_state:
            st.session_state.views = {}
        products = ["Apple", "Banana", "Carrot", "Dates"]
        prod = st.selectbox("View a Product", products)
        if st.button("View Product"):
            st.session_state.views[prod] = st.session_state.views.get(prod, 0) + 1
            st.success(f"Viewed {prod}!")

        st.subheader("Product View Counts")
        for p, v in st.session_state.views.items():
            st.write(f"{p}: {v} views")

        if st.session_state.views:
            recommended = max(st.session_state.views, key=st.session_state.views.get)
            st.info(f"Recommended product: **{recommended}**")

# 14. Track Skipped Products and Use View Time for Reports (Demo)
    elif task == "Track Skipped Products and Use View Time for Reports (Demo)":
        st.write("This demo tracks how long user views each product.")
        import time

        if 'start_time' not in st.session_state:
            st.session_state.start_time = time.time()
        if 'view_times' not in st.session_state:
            st.session_state.view_times = {}

        products = ["Item 1", "Item 2", "Item 3"]
        selected = st.selectbox("Select Product", products)
        current_time = time.time()
        prev_product = st.session_state.get('current_product', None)

    # Save view time for previous product
        if prev_product and prev_product != selected:
            duration = current_time - st.session_state.start_time
            st.session_state.view_times[prev_product] = st.session_state.view_times.get(prev_product, 0) + duration
            st.session_state.start_time = current_time

        st.session_state.current_product = selected

        st.write(f"Viewing **{selected}**")

        if st.session_state.view_times:
            st.subheader("View Times (seconds)")
            for p, t in st.session_state.view_times.items():
                st.write(f"{p}: {round(t, 2)} seconds")

# 15. Get IP Address and Location on Click
    elif task == "Get IP Address and Location on Click":
        st.write("Click button to fetch your IP and approximate location.")
        if st.button("Get IP & Location"):
            import requests
            try:
                r = requests.get("https://ipinfo.io/json")
                data = r.json()
                st.write(f"IP: {data.get('ip')}")
                loc = data.get('loc')
                city = data.get('city')
                region = data.get('region')
                country = data.get('country')
                st.write(f"Location: {city}, {region}, {country} (Coords: {loc})")
            except Exception as e:
                st.error(f"Failed to get IP info: {e}")

if menu=="🤖 Machine Learning Tasks":
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.linear_model import LinearRegression
    import paramiko

    st.title("ML Imputation Tasks & MongoDB Docker Runner")

    task = st.selectbox("Select Task", [
        "Q1: Explore Imputation Techniques and Analyze",
        "Q2: Linear Regression Imputation",
        "Run MongoDB Server in Docker"
    ])

# --- ML Dataset Upload ---
    if task in ["Q1: Explore Imputation Techniques and Analyze", "Q2: Linear Regression Imputation"]:
        uploaded_file = st.file_uploader("Upload CSV dataset (must have 'Y' column with missing values)", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset preview:")
            st.dataframe(df.head())
            target_col = st.selectbox("Select target column (Y)", options=df.columns, index=0)
            missing_count = df[target_col].isna().sum()


# --- Q1: Explore and Analyze ---
    if task == "Q1: Explore Imputation Techniques and Analyze" and uploaded_file:
        st.subheader("Missing Value Imputation Exploration")

        missing_count = df['Y'].isna().sum()
        total = len(df)
        st.write(f"Missing values in Y: {missing_count} / {total} ({missing_count/total*100:.2f}%)")

    # Mean Imputation
        mean_imp = SimpleImputer(strategy='mean').fit_transform(df[['Y']])
        st.write("Sample Mean Imputation for Y:", mean_imp[:5].flatten())

    # Median Imputation
        median_imp = SimpleImputer(strategy='median').fit_transform(df[['Y']])
        st.write("Sample Median Imputation for Y:", median_imp[:5].flatten())

    # Mode Imputation
        mode_imp = SimpleImputer(strategy='most_frequent').fit_transform(df[['Y']])
        st.write("Sample Mode Imputation for Y:", mode_imp[:5].flatten())

    # KNN Imputation on full dataset
        st.write("Running KNN Imputation (k=3) on entire dataset, please wait...")
        knn_imp = KNNImputer(n_neighbors=3).fit_transform(df)
        y_index = df.columns.get_loc('Y')
        st.write("Sample KNN Imputed Y values:", knn_imp[:5, y_index])

        st.markdown("""
        ### Analysis
        - If 'Y' correlates strongly with other features, Linear Regression can be effective for imputation.
        - Otherwise, simpler methods (mean, median, mode) might suffice.
        - KNN imputation considers feature similarity and may perform better when relationships are nonlinear.
        - Linear Regression assumes linear relationships and requires sufficient non-missing data.
        """)

# --- Q2: Linear Regression Imputation ---
    if task == "Q2: Linear Regression Imputation" and uploaded_file:
        st.subheader("Linear Regression to Impute Missing Y Values")

    # Split data into known Y and missing Y
        known = df[df['Y'].notna()]
        unknown = df[df['Y'].isna()]

        if unknown.empty:
            st.info("No missing values detected in 'Y'.")
        else:
            X_train = known.drop(columns=['Y'])
            y_train = known['Y']
            X_pred = unknown.drop(columns=['Y'])

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_pred)

            df.loc[df['Y'].isna(), 'Y'] = y_pred

            st.success("Missing values in 'Y' have been imputed using Linear Regression.")
            st.write("Updated Dataset:")
            st.dataframe(df)

# --- MongoDB Docker Run ---
    if task == "Run MongoDB Server in Docker":
        st.subheader("Run MongoDB Docker Container on Remote Machine")

        ip = st.text_input("Remote SSH IP")
        username = st.text_input("SSH Username", value="root")
        password = st.text_input("SSH Password", type="password")
        run_btn = st.button("Start MongoDB Docker Container")

        if run_btn:
            if not ip or not username or not password:
                st.error("Please provide all SSH credentials.")
            else:
                try:
                    client = paramiko.SSHClient()
                    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                    client.connect(ip, username=username, password=password)

                # Pull and run MongoDB container commands
                    commands = [
                        "docker pull mongo",
                        "docker run -d -p 27017:27017 --name mongodb mongo"
                    ]

                    for cmd in commands:
                        stdin, stdout, stderr = client.exec_command(cmd)
                        out = stdout.read().decode()
                        err = stderr.read().decode()
                        if out:
                            st.text(f"Output of '{cmd}':\n{out}")
                        if err:
                            st.error(f"Error from '{cmd}':\n{err}")

                    st.success("MongoDB container started successfully!")
                    client.close()
                except Exception as e:
                    st.error(f"SSH/Docker command failed: {e}")


# ------------------ Twilio Call ------------------
if menu == "📞 Twilio Call":
    acc = st.text_input("Account SID")
    tok = st.text_input("Auth Token", type="password")
    from_num = st.text_input("Twilio Number")
    to_num = st.text_input("To Number")
    if st.button("Call"):
        try:
            client = Client(acc, tok)
            call = client.calls.create(
                twiml='<Response><Say>Hi! This is a test call.</Say></Response>',
                to=to_num, from_=from_num)
            st.success(f"Call started: {call.sid}")
        except Exception as e:
            st.error(str(e))
# ------------------ Send SMS ------------------
if menu == "💬 Send SMS":
    acc = st.text_input("Account SID")
    tok = st.text_input("Auth Token", type="password")
    from_num = st.text_input("Twilio Number")
    to_num = st.text_input("To Number")
    msg = st.text_area("Message")
    if st.button("Send SMS"):
        try:
            client = Client(acc, tok)
            message = client.messages.create(body=msg, from_=from_num, to=to_num)
            st.success(f"Message SID: {message.sid}")
        except Exception as e:
            st.error(str(e))
# ------------------ WhatsApp Message ------------------
import threading
import pyautogui
import time
if menu == "🟢 WhatsApp Message":
    num = st.text_input("Enter number with country code")
    msg = st.text_area("Message")
    wait_time = st.slider("Wait time (seconds)", 5, 20, 10)
    def send_msg():
        try:
            pw.sendwhatmsg_instantly(num, msg, wait_time=wait_time, tab_close=False)
            time.sleep(wait_time + 5)  # wait extra for page to fully load
            pyautogui.press("enter")  # simulate Enter key to send
        except Exception as e:
            st.error(str(e))

    if st.button("Send Message"):
        threading.Thread(target=send_msg).start()
        st.success("Message is being sent. Do not move your mouse or change window.")
# ------------------ Send Email (pywhatkit) ------------------
elif menu == "📧 Send Email (pywhatkit)":
    st.title("📧 Send Email using pywhatkit")

    from_email = st.text_input("Sender Email (Gmail only)")
    password = st.text_input("App Password", type="password")
    to_email = st.text_input("Receiver Email")
    subject = st.text_input("Subject")
    message = st.text_area("Message")

    if st.button("Send Email"):
        sender = from_email.strip()
        receiver = to_email.strip()
        subj = subject.strip()
        msg = message.strip()

        if not (sender and password and receiver and subj and msg):
            st.warning("⚠️ Please fill all the fields.")
        elif "@" not in receiver:
            st.error("❌ Invalid email address.")
        else:
            try:
                # ✅ Make sure this order is correct
                pw.send_mail(sender, password,subj, msg,receiver )
                st.success("✅ Email sent successfully!")
            except Exception as e:
                st.error(f"❌ Failed to send email:\n{str(e)}")

#zrks apfb yjsi kxuc
# ------------------ Post on LinkedIn ------------------
elif menu == "🔗 Post on LinkedIn":
    st.subheader("📢 Post to LinkedIn via API")

    access_token = st.text_input("🔑 Access Token", type="password")
    author_urn = st.text_input("🧾 Author URN (e.g., urn:li:person:xxxx)")
    post_text = st.text_area("📝 Your Post Text")

    if st.button("🚀 Post to LinkedIn"):
        if not access_token or not author_urn or not post_text:
            st.warning("⚠️ Please fill in all the fields.")
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Restli-Protocol-Version": "2.0.0"
                }

                payload = {
                    "author": author_urn,
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {
                                "text": post_text
                            },
                            "shareMediaCategory": "NONE"
                        }
                    },
                    "visibility": {
                        "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
                    }
                }

                response = requests.post(
                    "https://api.linkedin.com/v2/ugcPosts",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    st.success("✅ Post published successfully!")
                else:
                    st.error(f"❌ Failed with status: {response.status_code}")
                    st.json(response.json())

            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")


# ------------------ Draw Grid Image ------------------
from streamlit_drawable_canvas import st_canvas
import streamlit as st

if menu == "🎨 Draw Grid Image":
    st.subheader("🧑‍🎨 Draw on Grid Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # No fill
        stroke_width=2,
        stroke_color="#00FFAA",
        background_color="#000000",
        height=600,
        width=800,
        drawing_mode="freedraw",
        key="draw_canvas"
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption="🖼️ Your Drawing")

    st.info("Use your mouse to draw. To erase, refresh the page.")
# ------------------ Face Swap ------------------
if menu == "🔄 Face Swap":
    import cv2
    import numpy as np

    st.subheader("🔄 Face Swap with Webcam + Preview")

    def capture_image():
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return frame

    def detect_and_draw(img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img, faces

    def face_swap_images(img1, img2, faces1, faces2):
        if len(faces1) == 0 or len(faces2) == 0:
            return None, None

        x1, y1, w1, h1 = faces1[0]
        x2, y2, w2, h2 = faces2[0]

        face1 = cv2.resize(img1[y1:y1+h1, x1:x1+w1], (w2, h2))
        face2 = cv2.resize(img2[y2:y2+h2, x2:x2+w2], (w1, h1))

        img1[y1:y1+h1, x1:x1+w1] = face2
        img2[y2:y2+h2, x2:x2+w2] = face1

        path1 = "swapped1.png"
        path2 = "swapped2.png"
        cv2.imwrite(path1, img1)
        cv2.imwrite(path2, img2)

        return path1, path2

    # Image session states
    if "image1" not in st.session_state: st.session_state["image1"] = None
    if "image2" not in st.session_state: st.session_state["image2"] = None
    if "faces1" not in st.session_state: st.session_state["faces1"] = []
    if "faces2" not in st.session_state: st.session_state["faces2"] = []

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Capture First Image", key="capture1"):
            img = capture_image()
            if img is not None:
                img_drawn, faces = detect_and_draw(img.copy())
                st.session_state["image1"] = img
                st.session_state["faces1"] = faces
                st.image(img_drawn, caption=f"✅ First Image ({len(faces)} face(s))", channels="BGR")

    with col2:
        if st.button("📸 Capture Second Image", key="capture2"):
            img = capture_image()
            if img is not None:
                img_drawn, faces = detect_and_draw(img.copy())
                st.session_state["image2"] = img
                st.session_state["faces2"] = faces
                st.image(img_drawn, caption=f"✅ Second Image ({len(faces)} face(s))", channels="BGR")

    if st.session_state["image1"] is not None and st.session_state["image2"] is not None:
        if st.button("🔄 Swap Faces", key="swap_faces"):
            result1, result2 = face_swap_images(
                st.session_state["image1"].copy(),
                st.session_state["image2"].copy(),
                st.session_state["faces1"],
                st.session_state["faces2"]
            )
            if result1 and result2:
                st.success("✅ Face Swap Completed!")
                st.image(result1, caption="Swapped Image 1")
                st.image(result2, caption="Swapped Image 2")
            else:
                st.error("❌ Face(s) not detected in one or both images.")
# ------------------ Download Website HTML ------------------
def download_html(url):
    try:
        res = requests.get(url)
        with open("website.html", "w", encoding="utf-8") as f:
            f.write(res.text)
        return "website.html"
    except Exception as e:
        return str(e)
if menu == "🌍 Download Website HTML":
    url = st.text_input("Enter URL", "https://www.geeksforgeeks.org")
    if st.button("Download"):
        file = download_html(url)
        if os.path.exists(file):
            with open(file, encoding="utf-8") as f:
                st.code(f.read()[:1000])
        else:
            st.error(file)
# ------------------ Google Search ------------------
if menu == "🔎 Google Search":
    query = st.text_input("Search Query", "Python programming tutorials")
    if st.button("Search"):
        try:
            results = list(search(query, num_results=5))
            for url in results:
                st.write(url)
        except Exception as e:
            st.error(str(e))
# ------------------ Ping Flask API ------------------
if menu == "📡 Ping API":
    st.header("📡 Ping an API Endpoint")
    api_url = st.text_input("Enter API URL:", "http://127.0.0.1:5000/api/ping")
    if st.button("Ping API"):
        try:
            r = requests.get(api_url)
            try:
                st.json(r.json())  # Try rendering as JSON
            except:
                st.write(r.text)   # Fallback to plain text
        except Exception as e:
            st.error(str(e))
# ------------------ File Manager ------------------
import os
import shutil
import stat
import time

try:
    import pwd
    import grp
except ImportError:
    pwd = None
    grp = None

if menu == "🗂 File Manager":
    st.title("🗂 Guided File Manager")

    # Step 1: OS Selection
    os_mode = st.selectbox("🖥 Operating System", ["Windows", "Linux"])

    # Step 2: Task Selection
    task = st.selectbox("⚙ Task You Want to Perform", [
        "View Directory", "Rename", "Delete", "Move", "Create File", "Create Folder", "View File", "Change Directory"
    ])

    # Step 3: Directory Selection
    cwd = st.session_state.get("cwd", os.getcwd())
    new_dir = st.text_input("📁 Current Directory", value=cwd)
    if new_dir != cwd and os.path.isdir(new_dir):
        st.session_state["cwd"] = new_dir
        cwd = new_dir

    try:
        files = os.listdir(cwd)
    except Exception as e:
        st.error(f"Unable to read directory: {e}")
        files = []

    files = sorted(files)
    selected_item = st.selectbox("📂 Select a File/Folder", files) if files else None
    selected_path = os.path.join(cwd, selected_item) if selected_item else None

    # Show directory listing (if task is View Directory)
    if task == "View Directory":
        st.subheader("📄 Directory Content")
        show_hidden = st.checkbox("Show Hidden Files", value=False)
        visible_files = [f for f in files if show_hidden or not f.startswith('.')]

        for file in visible_files:
            path = os.path.join(cwd, file)
            try:
                stats = os.stat(path)
                mode = stat.filemode(stats.st_mode)
                size = stats.st_size
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats.st_mtime))

                if os_mode == "Linux" and pwd and grp:
                    try:
                        owner = pwd.getpwuid(stats.st_uid).pw_name
                        group = grp.getgrgid(stats.st_gid).gr_name
                    except:
                        owner = group = "unknown"
                else:
                    owner = group = "user"

                st.text(f"{mode} {owner}:{group} {size}B {mtime}  {file}")
            except Exception as e:
                st.text(f"[Error reading {file}]: {e}")

    # Rename
    elif task == "Rename" and selected_path:
        new_name = st.text_input("📝 New Name")
        if st.button("Rename"):
            try:
                os.rename(selected_path, os.path.join(cwd, new_name))
                st.success("Renamed successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Rename failed: {e}")

    # Delete
    elif task == "Delete" and selected_path:
        if st.button("Delete"):
            try:
                if os.path.isfile(selected_path):
                    os.remove(selected_path)
                else:
                    shutil.rmtree(selected_path)
                st.success("Deleted successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    # Move
    elif task == "Move" and selected_path:
        dest = st.text_input("📦 Destination Path")
        if st.button("Move"):
            try:
                shutil.move(selected_path, dest)
                st.success("Moved successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Move failed: {e}")

    # Create File
    elif task == "Create File":
        file_name = st.text_input("📄 New File Name")
        if st.button("Create File"):
            try:
                open(os.path.join(cwd, file_name), 'w').close()
                st.success("File created!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"File creation failed: {e}")

    # Create Folder
    elif task == "Create Folder":
        folder_name = st.text_input("📁 New Folder Name")
        if st.button("Create Folder"):
            try:
                os.makedirs(os.path.join(cwd, folder_name), exist_ok=True)
                st.success("Folder created!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Folder creation failed: {e}")

    # View File
    elif task == "View File" and selected_path:
        if os.path.isfile(selected_path) and selected_path.endswith(('.txt', '.py', '.log', '.md')):
            try:
                with open(selected_path, 'r', encoding='utf-8', errors='ignore') as f:
                    st.text_area("📖 File Content", f.read(), height=300)
            except Exception as e:
                st.error(f"Cannot read file: {e}")
        else:
            st.warning("Selected item is not a readable text file.")

    # Change Directory
    elif task == "Change Directory" and selected_path:
        if os.path.isdir(selected_path):
            if st.button("Enter Directory"):
                st.session_state["cwd"] = selected_path
                st.experimental_rerun()
        else:
            st.warning("Selected item is not a directory.")
import streamlit as st
import paramiko
import time

if menu=="🐳 Remote Docker Command Center":
    st.title("🐳 Remote Docker Command Center")

    ip = st.text_input("Remote SSH IP")
    username = st.text_input("SSH Username", value="root")
    password = st.text_input("SSH Password", type="password")
    connect_btn = st.button("Connect SSH")

    if "connected" not in st.session_state:
        st.session_state.connected = False

    if connect_btn and ip and username and password:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(ip, username=username, password=password)
            st.session_state.client = client
            st.session_state.connected = True
            st.success("✅ Connected to remote via SSH!")
        except Exception as e:
            st.session_state.connected = False
            st.error(f"SSH connection failed: {e}")

    def run_ssh_command(cmd):
        try:
            stdin, stdout, stderr = st.session_state.client.exec_command(cmd, get_pty=True)
            out = stdout.read().decode()
            err = stderr.read().decode()
            return out.strip(), err.strip()
        except Exception as e:
            return "", f"Exception running command: {e}"

    def check_remote_dir(path):
        out, err = run_ssh_command(f'test -d "{path}" && echo "EXISTS" || echo "MISSING"')
        return out == "EXISTS"

    def create_remote_dir(path):
        out, err = run_ssh_command(f'mkdir -p "{path}"')
        return (err == "")

    def write_remote_file(path, content):
        # Escape quotes and special chars
        escaped = content.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
        cmd = f'echo "{escaped}" > "{path}"'
        out, err = run_ssh_command(cmd)
        return (err == "")

    def ensure_linear_context(path):
        if not check_remote_dir(path):
            st.info(f"Creating linear regression context folder at {path}")
            create_remote_dir(path)
            dockerfile = """FROM python:3.9-slim
WORKDIR /app
COPY linear.py .
RUN pip install scikit-learn numpy
CMD ["python", "linear.py"]
"""
            linear_py = """
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression().fit(X, y)
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
"""
            write_remote_file(f"{path}/Dockerfile", dockerfile)
            write_remote_file(f"{path}/linear.py", linear_py)
            st.success(f"Linear regression context created at {path}")

    def ensure_flask_context(path):
        if not check_remote_dir(path):
            st.info(f"Creating Flask app context folder at {path}")
            create_remote_dir(path)
            dockerfile = """FROM python:3.9-slim
WORKDIR /app
COPY app.py .
RUN pip install flask
EXPOSE 5000
CMD ["python", "app.py"]
"""
            app_py = """
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello from Flask inside Docker!"
if __name__ == '__main__':
    app.run(host='0.0.0.0')
"""
            write_remote_file(f"{path}/Dockerfile", dockerfile)
            write_remote_file(f"{path}/app.py", app_py)
            st.success(f"Flask app context created at {path}")

    def ensure_firefox_context(path):
        if not check_remote_dir(path):
            st.info(f"Creating Firefox context folder at {path}")
            create_remote_dir(path)
            dockerfile = """FROM ubuntu:20.04
RUN apt-get update && apt-get install -y firefox
CMD ["firefox"]
"""
            write_remote_file(f"{path}/Dockerfile", dockerfile)
            st.success(f"Firefox context created at {path}")

# Default remote paths (you can customize)
    DEFAULT_LINEAR_DIR = "/home/user/projects/linear"
    DEFAULT_FLASK_DIR = "/home/user/projects/flask"
    DEFAULT_FIREFOX_DIR = "/home/user/projects/firefox"

    if st.session_state.connected:
        st.subheader("📦 Docker Operations")

        docker_ops = [
            "Show Docker Version",
            "List Docker Images",
            "List Running Containers",
            "List All Containers",
            "Pull an Image",
            "Build Linear Regression Image",
            "Run Linear Regression Container",
            "Build Flask App Image",
            "Run Flask App Container",
            "Run Python CLI Container",
            "Run Docker-in-Docker (DIND)",
            "Install Firefox in Container",
            "Play VLC in Container",
            "Setup Apache Server Container",
            "Run MongoDB Server Container",
            "Stop a Container",
            "Remove a Container"
        ]

        op = st.selectbox("Choose Docker Operation", docker_ops)
        output = ""
        err = ""

        def docker_image_exists(image_name):
            out, err = run_ssh_command(f'docker images -q {image_name}')
            return out != ""

        def build_image_if_missing(tag, context_path=None, ensure_context_fn=None):
            if ensure_context_fn and context_path:
                ensure_context_fn(context_path)
            if not docker_image_exists(tag):
                st.info(f"Image '{tag}' not found. Building now...")
                cmd = f"docker build -t {tag} {context_path}" if context_path else f"docker build -t {tag} ."
                out, err = run_ssh_command(cmd)
                return out, err
            else:
                return "", ""

        if op == "Show Docker Version":
            output, err = run_ssh_command("docker --version")

        elif op == "List Docker Images":
            output, err = run_ssh_command("docker images")

        elif op == "List Running Containers":
            output, err = run_ssh_command("docker ps")

        elif op == "List All Containers":
            output, err = run_ssh_command("docker ps -a")

        elif op == "Pull an Image":
            img_name = st.text_input("Image name to pull (e.g. python:3.9-slim)")
            if st.button("Pull Image") and img_name.strip():
                output, err = run_ssh_command(f"docker pull {img_name.strip()}")

        elif op == "Build Linear Regression Image":
            linear_dir = st.text_input("Path to Linear Regression Docker context", value=DEFAULT_LINEAR_DIR)
            tag = st.text_input("Image tag", value="linear_model")
            if st.button("Build Linear Regression Image"):
                ensure_linear_context(linear_dir)
                output, err = run_ssh_command(f"docker build -t {tag} {linear_dir}")

        elif op == "Run Linear Regression Container":
            tag = st.text_input("Image name", value="linear_model")
            if st.button("Run Container"):
                out_build, err_build = build_image_if_missing(tag, DEFAULT_LINEAR_DIR, ensure_linear_context)
                if err_build:
                    st.error(f"Build failed:\n{err_build}")
                else:
                    cmd = f"docker run --rm {tag}"
                    output, err = run_ssh_command(cmd)

        elif op == "Build Flask App Image":
            flask_dir = st.text_input("Path to Flask App Docker context", value=DEFAULT_FLASK_DIR)
            tag = st.text_input("Image tag", value="flask_app")
            if st.button("Build Flask Image"):
                ensure_flask_context(flask_dir)
                output, err = run_ssh_command(f"docker build -t {tag} {flask_dir}")

        elif op == "Run Flask App Container":
            tag = st.text_input("Image name", value="flask_app")
            port = st.text_input("Port mapping", value="5000:5000")
            if st.button("Run Flask Container"):
                out_build, err_build = build_image_if_missing(tag, DEFAULT_FLASK_DIR, ensure_flask_context)
                if err_build:
                    st.error(f"Build failed:\n{err_build}")
                else:
                    cmd = f"docker run -d -p {port} {tag}"
                    output, err = run_ssh_command(cmd)

        elif op == "Run Python CLI Container":
            tag = st.text_input("Image name")
            if st.button("Run Python CLI Container"):
                # No build context here; user must provide image
                if not docker_image_exists(tag):
                    st.error("Image not found locally. Please pull or build the image first.")
                else:
                    cmd = f"docker run --rm {tag}"
                output, err = run_ssh_command(cmd)

        elif op == "Run Docker-in-Docker (DIND)":
            if st.button("Run DIND Container"):
                cmd = "docker run --privileged -d docker:dind"
                output, err = run_ssh_command(cmd)

        elif op == "Install Firefox in Container":
            firefox_dir = st.text_input("Path to Firefox Docker context", value=DEFAULT_FIREFOX_DIR)
            tag = st.text_input("Image tag", value="firefox_app")
            build_btn = st.button("Build Firefox Image")
            run_btn = st.button("Run Firefox Container")
            if build_btn:
                ensure_firefox_context(firefox_dir)
                output, err = run_ssh_command(f"docker build -t {tag} {firefox_dir}")
            if run_btn:
                out_build, err_build = build_image_if_missing(tag, firefox_dir, ensure_firefox_context)
                if err_build:
                    st.error(f"Build failed:\n{err_build}")
                else:
                    cmd = f"docker run --rm -it {tag} firefox"
                    output, err = run_ssh_command(cmd)

        elif op == "Play VLC in Container":
            default_vlc_image = "jess/vlc"
            img = st.text_input("VLC Docker Image", value=default_vlc_image)
            if st.button("Run VLC Container"):
                cmd = f"docker run --rm -it --device /dev/snd {img} vlc"
                output, err = run_ssh_command(cmd)

        elif op == "Setup Apache Server Container":
            default_apache_image = "httpd"
            port = st.text_input("Port mapping", value="80:80")
            img = st.text_input("Apache Docker Image", value=default_apache_image)
            if st.button("Run Apache Container"):
                cmd = f"docker run -d -p {port} {img}"
                output, err = run_ssh_command(cmd)

        elif op == "Run MongoDB Server Container":
            default_mongo_image = "mongo"
            port = st.text_input("Port mapping", value="27017:27017")
            img = st.text_input("MongoDB Docker Image", value=default_mongo_image)
            if st.button("Run MongoDB Container"):
                cmd = f"docker run -d -p {port} {img}"
                output, err = run_ssh_command(cmd)

        elif op == "Stop a Container":
            cid = st.text_input("Container ID or name to stop")
            if st.button("Stop Container") and cid.strip():
                output, err = run_ssh_command(f"docker stop {cid.strip()}")

        elif op == "Remove a Container":
            cid = st.text_input("Container ID or name to remove")
            if st.button("Remove Container") and cid.strip():
                output, err = run_ssh_command(f"docker rm {cid.strip()}")

        else:
            output = "Select an operation and provide inputs as needed."

        if output:
            st.text_area("Output", output, height=300)
        if err:
            st.error(f"Error: {err}")

    else:
        st.warning("Please connect to SSH first.")

    
if menu == "🔐 Linux Command Executor":
    st.subheader("🔐 Linux Command Center")

    ssh_host = st.text_input("Host (IP or domain)")
    ssh_port = st.number_input("Port", value=22)
    ssh_user = st.text_input("Username")
    ssh_pass = st.text_input("Password", type="password")

    # Predefined 50 commands
    commands = [
        "ls", "pwd", "whoami", "uptime", "df -h", "free -m", "top -b -n1", "ps aux",
        "cat /etc/os-release", "uname -a", "netstat -tuln", "ifconfig", "ip a", "ping -c 4 google.com",
        "history", "du -sh *", "find / -type f -name '*.log'", "journalctl -xe", "tail -n 100 /var/log/syslog",
        "date", "cal", "env", "echo $PATH", "groups", "id", "lsblk", "mount", "df -i", "uptime -p", "who",
        "last", "hostname", "ls -alh", "crontab -l", "cat /etc/passwd", "cat /etc/group", "ss -tuln", "ip r",
        "iptables -L", "firewalld --state", "nmcli dev status", "systemctl list-units --type=service",
        "systemctl status sshd", "reboot", "shutdown now", "logout", "clear", "echo Hello from SSH", "ls /home"
    ]

    if st.button("Connect & Show Options"):
        if not (ssh_host and ssh_user and ssh_pass):
            st.warning("Please fill all SSH fields.")
        else:
            st.session_state["ssh_ready"] = True

    if st.session_state.get("ssh_ready"):
        st.success("SSH connected! Choose a command to run:")

        # Display numbered command list
        st.markdown("### 🔢 Predefined Command List")
        for i, cmd in enumerate(commands, 1):
            st.text(f"{i}. {cmd}")

        col1, col2 = st.columns(2)

        with col1:
            cmd_num = st.number_input("Run Command No. (1–50)", min_value=1, max_value=50, step=1)
            run_num_cmd = st.button("Run Selected Command")

        with col2:
            custom_cmd = st.text_input("Or Enter Your Own Command")
            run_custom_cmd = st.button("Run Custom Command")

        try:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=ssh_host, port=ssh_port, username=ssh_user, password=ssh_pass)

            if run_num_cmd:
                selected_command = commands[cmd_num - 1]
                st.info(f"Running command #{cmd_num}: `{selected_command}`")
                stdin, stdout, stderr = ssh.exec_command(selected_command)
            elif run_custom_cmd and custom_cmd.strip():
                st.info(f"Running your custom command: `{custom_cmd}`")
                stdin, stdout, stderr = ssh.exec_command(custom_cmd)
            else:
                stdin = stdout = stderr = None

            if stdout:
                st.code(stdout.read().decode())
            if stderr:
                err = stderr.read().decode()
                if err:
                    st.error(err)

            ssh.close()
        except Exception as e:
            st.error(f"SSH Error: {str(e)}")
#--------------Remotely access File Manager Windows-----------
if menu == "🧠 Remote File Control (Cross OS)":
    import requests

    st.title("🧠 Aryan's Remote File Manager")

    # === Agent Script (Remote PC File Manager) ===
    agent_code = '''
import os
import shutil
import socket
import platform
from flask import Flask, request, jsonify

app = Flask(__name__)
current_os = platform.system()
cwd = os.getcwd()

@app.route('/')
def home():
    return f"🖥 Agent running on {socket.gethostname()} ({current_os})"

@app.route('/list', methods=['GET'])
def list_dir():
    global cwd
    path = request.args.get('path', cwd)
    try:
        cwd = path
        files = os.listdir(path)
        return jsonify({"success": True, "cwd": cwd, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/rename', methods=['POST'])
def rename_file():
    data = request.json
    try:
        os.rename(data['old_path'], data['new_path'])
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete', methods=['POST'])
def delete_file():
    data = request.json
    try:
        if os.path.isfile(data['path']):
            os.remove(data['path'])
        else:
            shutil.rmtree(data['path'])
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/move', methods=['POST'])
def move_file():
    data = request.json
    try:
        shutil.move(data['source'], data['destination'])
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/create_file', methods=['POST'])
def create_file():
    data = request.json
    try:
        open(data['path'], 'w').close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/create_folder', methods=['POST'])
def create_folder():
    data = request.json
    try:
        os.makedirs(data['path'], exist_ok=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/view_file', methods=['GET'])
def view_file():
    path = request.args.get('path')
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return jsonify({"success": True, "content": content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/change_directory', methods=['POST'])
def change_directory():
    global cwd
    data = request.json
    try:
        os.chdir(data['path'])
        cwd = os.getcwd()
        return jsonify({"success": True, "cwd": cwd})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print(f"✅ Agent running on {current_os}")
    print("🌐 Listening on port 7860 (use your private IP address)")
    app.run(host='0.0.0.0', port=7860)
'''

    st.info("💡 Step 1: Ask your friend to run the agent script using `python remote_agent.py` on their system.")
    st.download_button("⬇️ Download Remote Agent Script", agent_code, file_name="remote_agent.py")

    st.markdown("---")
    st.header("🔌 Step 2: Connect to Remote PC")

    remote_ip = st.text_input("Enter Friend's IP Address", "127.0.0.1")
    base_url = f"http://{remote_ip}:7860"

    try:
        ping = requests.get(f"{base_url}/")
        st.success("✅ Connected to: " + ping.text)
    except:
        st.warning("⚠️ Could not connect. Ensure agent is running on remote PC.")

    task = st.selectbox("⚙️ Choose Task", [
        "View Directory", "Rename", "Delete", "Move",
        "Create File", "Create Folder", "View File", "Change Directory"
    ])

    st.markdown("### 🔧 Task Controls")

    if task == "View Directory":
        path = st.text_input("Directory to View", "/")
        if st.button("View"):
            r = requests.get(f"{base_url}/list", params={"path": path})
            data = r.json()
            if data["success"]:
                st.success(f"📁 Contents of: {data['cwd']}")
                for f in data["files"]:
                    st.text(f)
            else:
                st.error(data["error"])

    elif task == "Rename":
        old = st.text_input("Old Path")
        new = st.text_input("New Path")
        if st.button("Rename"):
            r = requests.post(f"{base_url}/rename", json={"old_path": old, "new_path": new})
            st.success("✅ Renamed!") if r.json()["success"] else st.error(r.json()["error"])

    elif task == "Delete":
        path = st.text_input("Path to Delete")
        if st.button("Delete"):
            r = requests.post(f"{base_url}/delete", json={"path": path})
            st.success("✅ Deleted!") if r.json()["success"] else st.error(r.json()["error"])

    elif task == "Move":
        source = st.text_input("Source Path")
        dest = st.text_input("Destination Path")
        if st.button("Move"):
            r = requests.post(f"{base_url}/move", json={"source": source, "destination": dest})
            st.success("✅ Moved!") if r.json()["success"] else st.error(r.json()["error"])

    elif task == "Create File":
        file_path = st.text_input("Full File Path to Create")
        if st.button("Create File"):
            r = requests.post(f"{base_url}/create_file", json={"path": file_path})
            st.success("✅ File Created!") if r.json()["success"] else st.error(r.json()["error"])

    elif task == "Create Folder":
        folder_path = st.text_input("Folder Path to Create")
        if st.button("Create Folder"):
            r = requests.post(f"{base_url}/create_folder", json={"path": folder_path})
            st.success("✅ Folder Created!") if r.json()["success"] else st.error(r.json()["error"])

    elif task == "View File":
        file_path = st.text_input("File Path to View")
        if st.button("View File"):
            r = requests.get(f"{base_url}/view_file", params={"path": file_path})
            data = r.json()
            if data["success"]:
                st.text_area("📖 File Content", data["content"], height=300)
            else:
                st.error(data["error"])

    elif task == "Change Directory":
        new_dir = st.text_input("Directory Path to Switch To")
        if st.button("Change Directory"):
            r = requests.post(f"{base_url}/change_directory", json={"path": new_dir})
            data = r.json()
            if data["success"]:
                st.success(f"✅ Changed to: {data['cwd']}")
            else:
                st.error(data["error"])

# ------------------ Run Flask in Background ------------------
def run_flask():
    app.run(port=5000)

threading.Thread(target=run_flask, daemon=True).start()
