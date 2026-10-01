import streamlit as st
import requests
import json
import os

API_BASE_URL = os.environ.get("SPA_API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Smart Photo Archive Search", layout="wide", page_icon="📸")


def fetch_stats():
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=3)
        return response.json()
    except Exception:
        return {}


def fetch_taxonomy_categories():
    try:
        response = requests.get(f"{API_BASE_URL}/taxonomy/categories", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def fetch_taxonomy_projects():
    try:
        response = requests.get(f"{API_BASE_URL}/taxonomy/projects", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def search_images(query, mode, category=None, project=None, label_type=None, top_k=50):
    params = {"q": query, "mode": mode.lower(), "top_k": top_k}
    if category and category != "All Categories":
        params["category"] = category
    if project and project != "All Projects":
        params["project"] = project
    if label_type and label_type != "All":
        params["label_type"] = label_type.lower()

    try:
        response = requests.get(f"{API_BASE_URL}/search", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"API Connection Error: {e}")
        return None


def main():
    st.title("🏛️ Smart Photo Archive Search")
    st.markdown("Search across institutional archives using natural language, open-vocabulary object tags, facial recognition, and IIHS domain taxonomy.")

    # Sidebar: Search Controls and Taxonomy Filters
    st.sidebar.header("🔍 Search Controls")
    mode = st.sidebar.selectbox("Search Mode", ["Hybrid", "Semantic", "Keyword", "Face"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Institutional Taxonomy Filters")
    
    categories = fetch_taxonomy_categories()
    selected_category = st.sidebar.selectbox("Primary Category", ["All Categories"] + categories)

    projects = fetch_taxonomy_projects()
    selected_project = st.sidebar.selectbox("Project / Event", ["All Projects"] + projects)

    selected_label_type = st.sidebar.selectbox("Taxonomy Label Type", ["All", "Object", "Context", "Scene"])

    st.sidebar.markdown("---")
    stats = fetch_stats()
    if stats:
        st.sidebar.subheader("📊 Archive Statistics")
        st.sidebar.metric("Total Images", f"{stats.get('total_images', 0):,}")
        st.sidebar.metric("Face Clusters", f"{stats.get('total_clusters', 0):,}")
        st.sidebar.metric("Tags Generated", f"{stats.get('unique_tags', 0):,}")
        st.sidebar.metric("Taxonomy Keywords", f"{stats.get('total_institutional_keywords', 0):,}")

    # Search Box & Trigger
    if mode in ["Hybrid", "Semantic", "Keyword"]:
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Search Archive",
                placeholder="e.g. 'solar panels rooftop', 'flooding during monsoon', 'community meeting', or 'convocation'",
                label_visibility="collapsed"
            )
        with col2:
            search_clicked = st.button("Search", type="primary", use_container_width=True)

        if search_clicked or query:
            if query:
                with st.spinner("Searching archive across multi-modal indices..."):
                    results_data = search_images(
                        query, mode,
                        category=selected_category,
                        project=selected_project,
                        label_type=selected_label_type
                    )
                    display_results(results_data)
            else:
                st.warning("Please enter a query or select an institutional filter.")

    elif mode == "Face":
        uploaded_file = st.file_uploader("Upload an image containing a face", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None and st.button("Search by Face", type="primary"):
            with st.spinner("Detecting face & matching ArcFace vector index..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {}
                    if selected_category != "All Categories":
                        params["category"] = selected_category
                    if selected_project != "All Projects":
                        params["project"] = selected_project

                    response = requests.post(f"{API_BASE_URL}/search/face", files=files, params=params, timeout=15)
                    if response.status_code == 200:
                        display_results(response.json())
                    else:
                        st.error("Face search failed.")
                except Exception as e:
                    st.error(f"API Connection Error: {e}")


def display_results(results_data):
    if not results_data or "results" not in results_data:
        st.info("No matching images found.")
        return

    results = results_data["results"]
    st.success(f"Found **{results_data.get('total_count', 0)}** results in **{results_data.get('search_time_ms', 0):.1f} ms** (mode: {results_data.get('mode')}).")

    cols = st.columns(4)
    for idx, res in enumerate(results):
        col = cols[idx % 4]
        with col:
            thumb_url = f"{API_BASE_URL.replace('/api', '')}{res['thumbnail_url']}"
            try:
                st.image(thumb_url, use_column_width=True)
            except Exception:
                st.markdown(f"🖼️ `Image #{res['image_id']}`")

            st.caption(f"Relevance: {res['score']:.3f}")
            
            # Show institutional tags if present
            if res.get('institutional_tags'):
                top_inst = [t['keyword'] for t in res['institutional_tags'][:2]]
                st.markdown(f"🏷️ **IIHS**: `{', '.join(top_inst)}`")
            elif res.get('tags'):
                st.markdown(f"Tags: *{', '.join(res['tags'][:3])}*")

            if res.get('caption'):
                st.markdown(f"*{res['caption'][:90]}...*")

            if st.button("Details", key=f"btn_{res['image_id']}"):
                view_image_detail(res['image_id'])


@st.dialog("Image Details")
def view_image_detail(image_id):
    try:
        resp = requests.get(f"{API_BASE_URL}/image/{image_id}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.subheader(f"Image ID: {image_id}")
            st.write(f"**Path**: `{data.get('file_path')}`")
            if data.get('caption'):
                st.markdown(f"**Description**: {data['caption']}")
            
            # Institutional Tags Section
            if data.get('institutional_tags'):
                st.markdown("### 🏛️ Institutional Taxonomy Tags")
                for it in data['institutional_tags']:
                    st.markdown(
                        f"- **{it['keyword']}** ({it.get('label_type', 'context')}) — "
                        f"Category: *{it.get('category')}* | Project: *{it.get('project')}* "
                        f"| Sim: `{it.get('similarity_score', 0):.2f}`"
                    )

            if data.get('tags'):
                st.markdown("### 🏷️ All Tags")
                st.write(", ".join([f"{t['tag']} ({t['source']})" for t in data['tags']]))

            if data.get('faces'):
                st.markdown(f"### 👤 Faces Detected: {len(data['faces'])}")

            st.json(data)
        else:
            st.error("Could not fetch image details.")
    except Exception as e:
        st.error(f"API Error: {e}")


if __name__ == "__main__":
    main()
