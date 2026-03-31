import streamlit as st
import requests
import platform
import psutil

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #2d3748;
    text-align: center;
}
.skill-chip {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
    font-size: 13px;
}
.skill-found {
    background-color: #064e3b;
    color: #34d399;
}
.skill-missing {
    background-color: #4c0519;
    color: #f87171;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("💻 System Telemetry")
    st.divider()

    st.subheader("Compute Engine")
    st.info(f"**Processor:** {platform.processor() or 'x86_64'}")
    st.info(f"**OS:** {platform.system()} {platform.release()}")

    mem = psutil.virtual_memory()
    st.subheader("Resource Usage")
    st.progress(mem.percent / 100)
    st.caption(
        f"{round(mem.used / (1024**3), 2)}GB / {round(mem.total / (1024**3), 2)}GB"
    )

    st.divider()
    st.caption("v4 Production ATS Engine Active")

# ---------------- API ----------------
@st.cache_data(show_spinner=False)
def get_job_taxonomy():
    try:
        res = requests.get("http://localhost:8000/jobs", timeout=5)
        res.raise_for_status()
        return res.json()
    except:
        return None

jobs_data = get_job_taxonomy()

if not jobs_data:
    st.error("🚨 Backend Unreachable.")
    st.stop()

job_map = {j.get("title"): j for j in jobs_data}

# ---------------- MAIN ----------------
st.title("🛡️ AI Resume Screening Center")
st.caption("Production-grade AI candidate evaluation system")

c1, c2 = st.columns(2)

with c1:
    selected_title = st.selectbox("🎯 Role", list(job_map.keys()))
    current_job = job_map.get(selected_title, {})

    with st.expander("Role Details"):
        st.write(current_job.get("industry_standard_summary", ""))
        st.caption(", ".join(current_job.get("technical_stack", [])))

with c2:
    uploaded_file = st.file_uploader("📄 Resume", type=["pdf", "docx"])

# ---------------- RUN ----------------
if st.button("🚀 Analyze Candidate"):

    if not uploaded_file:
        st.warning("Upload resume")
        st.stop()

    with st.spinner("Running AI analysis..."):
        try:
            res = requests.post(
                "http://localhost:8000/match",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                data={"job_id": current_job.get("role_id")},
                timeout=20
            )
            payload = res.json()
        except Exception as e:
            st.error(str(e))
            st.stop()

    analysis = payload.get("analysis", {})
    perf = payload.get("performance", {})

    score = analysis.get("match_percentage", 0)
    context = analysis.get("context_integrity", 0)
    exp = analysis.get("experience_score", 0)
    skill = analysis.get("skill_coverage", 0)

    # ---------------- TOP DASHBOARD ----------------
    st.markdown("## 📊 Candidate Evaluation")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Overall", f"{score}%")
    m2.metric("Experience", f"{exp}%")
    m3.metric("Skills", f"{skill}%")
    m4.metric("Context", f"{context}%")

    st.progress(score / 100)

    # Decision banner
    if score >= 75:
        st.success("🟢 Strong Hire Signal")
    elif score >= 50:
        st.warning("🟡 Consider / Needs Review")
    else:
        st.error("🔴 Weak Match")

    st.divider()

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs(["Skills Match", "Gaps", "Decision"])

    # -------- SKILLS FOUND --------
    with tab1:
        st.markdown("### ✅ Matching Skills")

        for skill in analysis.get("found_skills", []):
            st.markdown(
                f'<span class="skill-chip skill-found">{skill}</span>',
                unsafe_allow_html=True
            )

    # -------- MISSING --------
    with tab2:
        st.markdown("### ⚠️ Missing Skills")

        for skill in analysis.get("missing_skills", []):
            st.markdown(
                f'<span class="skill-chip skill-missing">{skill}</span>',
                unsafe_allow_html=True
            )

    # -------- DECISION --------
    with tab3:
        st.markdown("### 🧠 Hiring Recommendation")

        if score >= 75:
            st.markdown("**→ Move to technical / final round**")
            st.markdown("Candidate demonstrates strong alignment across skills and experience.")
        elif score >= 50:
            st.markdown("**→ Consider with reservations**")
            st.markdown("Some gaps present — may require upskilling or role adjustment.")
        else:
            st.markdown("**→ Not recommended**")
            st.markdown("Significant mismatch in core competencies and experience.")

        st.divider()

        st.caption(f"Latency: {perf.get('latency')} | RAM: {perf.get('ram_usage')}")