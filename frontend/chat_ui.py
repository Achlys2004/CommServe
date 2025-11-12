"""
CommServe Chat Interface - Conversational AI Mode
Natural chat experience with context-aware responses
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from datetime import datetime
from backend.query_engine import QueryEngine

# Page config
st.set_page_config(
    page_title="CommServe Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for improved UI
st.markdown(
    """
<style>
    /* Chat message styling */
    .user-message {
        background-color: #667eea;
        color: white;
        padding: 10px 14px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 75%;
        float: right;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
    }
    .assistant-message {
        background-color: #f0f2f6;
        color: #333;
        padding: 10px 14px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 75%;
        float: left;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
    }
    .message-container {
        overflow: auto;
        clear: both;
        margin-bottom: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background: #667eea;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-size: 14px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #5568d3;
        transform: translateY(-1px);
    }
    
    /* Commentary box */
    .commentary-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-left: 4px solid #ffd700;
        padding: 12px 16px;
        margin: 10px 0;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .commentary-box strong {
        color: #ffd700;
        font-weight: 600;
    }
    
    /* Feedback section */
    .feedback-container {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0;
        font-size: 13px;
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
    }
    
    /* Metric cards */
    .stMetric {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 8px;
    }
    
    /* Divider spacing */
    hr {
        margin: 12px 0 !important;
    }
    
    /* Example chips */
    div[data-testid="column"] button {
        font-size: 13px !important;
        padding: 6px 12px !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 40px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state with unique session ID per browser session
if "session_id" not in st.session_state:
    import uuid

    # Generate unique session ID - ensures fresh memory every time
    st.session_state.session_id = (
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    )
    st.session_state.session_initialized = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    # Initialize with orchestrator enabled for conversational mode
    # Use unique session ID for fresh cache per browser session (zero bias)
    st.session_state.engine = QueryEngine(
        use_orchestrator=True, session_id=st.session_state.session_id
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "👋 Hey! I'm Sci, your data analyst assistant. I can help you explore the Olist e-commerce dataset. Ask me anything about sales, products, customers, or reviews!",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )

# Initialize feedback state
if "pending_feedback" not in st.session_state:
    st.session_state.pending_feedback = {}

# Sidebar
with st.sidebar:
    st.header("💬 Chat Settings")

    # Session info
    session_id = st.session_state.engine.session_id
    st.caption(f"Session: `{session_id}`")

    # AI Provider Status
    st.subheader("🤖 AI Provider Status")
    try:
        from backend.llm_client import get_tier_status

        tier_status = get_tier_status()

        for tier, info in tier_status.items():
            status_emoji = (
                "✅"
                if info["available"] and not info["in_cooldown"]
                else "⏳" if info["in_cooldown"] else "❌"
            )
            provider_name = info["provider"].title()

            if info["in_cooldown"]:
                status_text = f"{status_emoji} {provider_name}: Cooldown ({info['cooldown_remaining_seconds']}s)"
            elif info["available"]:
                status_text = f"{status_emoji} {provider_name}: Active"
            else:
                status_text = f"{status_emoji} {provider_name}: Unavailable"

            st.caption(status_text)
    except Exception as e:
        st.caption("🤖 AI Status: Loading...")

    # Context summary
    st.subheader("📝 Recent Context")
    try:
        if (
            hasattr(st.session_state.engine, "_orchestrator")
            and st.session_state.engine._orchestrator
        ):
            context_summary = (
                st.session_state.engine._orchestrator.get_context_summary()
            )
            if context_summary != "No conversation history yet.":
                st.text(context_summary)
            else:
                st.info("No conversation yet")
        else:
            st.info("Context unavailable")
    except:
        st.info("Context unavailable")

    st.divider()

    # Quick actions
    st.subheader("⚡ Quick Actions")

    if st.button("🗑️ Clear Chat", width="stretch"):
        st.session_state.messages = [
            st.session_state.messages[0]
        ]  # Keep welcome message
        if (
            hasattr(st.session_state.engine, "_orchestrator")
            and st.session_state.engine._orchestrator
        ):
            st.session_state.engine._orchestrator.clear_conversation()
        st.rerun()

    if st.button("🔄 New Session", width="stretch"):
        import uuid

        # Generate completely new session ID with zero cache
        new_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        st.session_state.session_id = new_session_id
        st.session_state.engine = QueryEngine(
            session_id=new_session_id,
            use_orchestrator=True,
        )
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "🔄 New session started with fresh memory! What would you like to explore?",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        ]
        st.session_state.pending_feedback = {}
        st.rerun()

    st.divider()

    # Stats
    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Mode", "🤖 Conversational AI")

# Main chat interface
st.title("💬 CommServe Chat")
st.caption("Natural conversation with your data analyst AI")

# Example queries (as chips)
st.subheader("💡 Try asking:")
col1, col2, col3 = st.columns(3)

examples = [
    "Show me top 5 product categories by revenue",
    "What are the most loved products?",
    "Which products do customers hate?",
]

for i, example in enumerate(examples):
    with [col1, col2, col3][i]:
        if st.button(example, key=f"ex_{i}", width="stretch"):
            st.session_state.pending_query = example
            st.rerun()

st.divider()

# Chat messages display
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")

        if role == "user":
            st.markdown(
                f"""
            <div class="message-container">
                <div class="user-message">
                    <strong>You</strong> • {timestamp}<br/>
                    {content}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # Check if this is a system notice
            is_system_notice = content.startswith("[SYSTEM NOTICE]")

            if is_system_notice:
                st.markdown(
                    f"""
            <div class="message-container">
                <div class="assistant-message" style="background-color: #fff3cd; color: #856404; border-left: 4px solid #ffc107;">
                    <strong>⚠️ System Notice</strong> • {timestamp}<br/>
                    {content.replace("[SYSTEM NOTICE]", "").strip()}
                </div>
            </div>
            """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
            <div class="message-container">
                <div class="assistant-message">
                            <strong>Sci</strong> • {timestamp}<br/>
                            {content}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Show additional data if present
            if "result" in msg:
                result = msg["result"]

                # Show preprocessing warnings if any
                if result.get("warnings"):
                    for warning in result["warnings"]:
                        st.info(warning)

                # Show conversational summary/commentary if available for ALL result types
                if result.get("conversational_summary"):
                    # Process markdown bold syntax to HTML
                    summary_text = result["conversational_summary"]
                    # Convert **text** to <strong>text</strong>
                    import re

                    summary_text = re.sub(
                        r"\*\*(.*?)\*\*", r"<strong>\1</strong>", summary_text
                    )
                    # Convert newlines to <br> tags
                    summary_text = summary_text.replace(chr(10), "<br>").replace(
                        chr(13), ""
                    )

                    # Create a single styled container for insights with proper content containment
                    insight_html = f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-left: 5px solid #ffd700;
                        padding: 20px;
                        margin: 12px 0;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <div style="margin-bottom: 12px;">
                            <span style="color: #ffd700; font-size: 18px; font-weight: bold;">💡 AI Insight</span>
                        </div>
                        <div style="
                            background: rgba(255, 255, 255, 0.15);
                            border-left: 4px solid #ffd700;
                            padding: 16px;
                            border-radius: 6px;
                            line-height: 1.6;
                            font-size: 15px;
                        ">
                            {summary_text}
                        </div>
                    </div>
                    """
                    st.markdown(insight_html, unsafe_allow_html=True)

                # Show SQL results as table
                if result.get("sql_result") and result["sql_result"].get("rows"):
                    rows = result["sql_result"]["rows"]
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        height=min(400, len(rows) * 35 + 38),
                    )
                    st.caption(f"📊 {len(rows)} rows")

                # Show natural language summary if available (only if no conversational summary and not SQL)
                if (
                    result.get("natural_language_summary")
                    and not result.get("conversational_summary")
                    and result.get("type") != "sql"
                ):
                    st.info(result["natural_language_summary"])

                # Show code results if available
                if result.get("type") == "code" and result.get("execution_result"):
                    exec_result = result["execution_result"]
                    if exec_result.get("status") == "success":
                        # Show execution results with AI interpretation
                        output_type = exec_result.get("output_type", "text")

                        if output_type == "image" and exec_result.get("images"):
                            # Show images prominently for visualizations
                            st.subheader("📊 Data Visualization")
                            for i, img_data in enumerate(exec_result["images"]):
                                st.image(
                                    img_data,
                                    caption=f"Analysis Result {i+1}",
                                    use_container_width=True,
                                )
                            # Note: AI interpretation already shown above in commentary box

                        elif exec_result.get("stdout"):
                            # Show text results
                            st.subheader("📈 Analysis Results")
                            st.code(exec_result["stdout"], language="text")
                            # Note: AI interpretation already shown above in commentary box

                        if exec_result.get("has_visualization") and not exec_result.get(
                            "images"
                        ):
                            st.warning(
                                "⚠️ Visualization attempted but no plots were generated."
                            )

                    else:
                        st.error("❌ Analysis failed")
                        if exec_result.get("error"):
                            st.text_area(
                                "Error Details:", exec_result["error"], height=100
                            )

                # Add feedback system for assistant messages with results
                msg_idx = st.session_state.messages.index(msg)
                feedback_key = f"feedback_{msg_idx}"

                # Determine the detected query type from result
                detected_type = result.get("type", "unknown").upper()
                if detected_type == "SQL":
                    detected_label = "SQL Query"
                elif detected_type == "RAG":
                    detected_label = "RAG Search"
                elif detected_type == "CODE":
                    detected_label = "Python Analysis"
                elif detected_type == "METADATA":
                    detected_label = "Dataset Info"
                else:
                    detected_label = detected_type

                # Create feedback section
                with st.expander(
                    f"💬 Was this response correct? (Detected: {detected_label})",
                    expanded=False,
                ):
                    st.markdown(
                        '<div class="feedback-container">', unsafe_allow_html=True
                    )

                    col_fb1, col_fb2 = st.columns([3, 2])

                    with col_fb1:
                        st.write("**Help improve the system:**")
                        feedback_option = st.selectbox(
                            "If wrong, what should it have been?",
                            options=[
                                "✅ This is correct",
                                "Should be SQL Query",
                                "Should be RAG Search",
                                "Should be Python Analysis",
                                "Should be Dataset Info",
                            ],
                            key=f"feedback_select_{msg_idx}",
                            label_visibility="collapsed",
                        )

                    with col_fb2:
                        if st.button(
                            "Submit Feedback",
                            key=f"feedback_btn_{msg_idx}",
                            type="secondary",
                        ):
                            import requests

                            # Extract the user's original query
                            original_query = None
                            if (
                                msg_idx > 0
                                and st.session_state.messages[msg_idx - 1]["role"]
                                == "user"
                            ):
                                original_query = st.session_state.messages[msg_idx - 1][
                                    "content"
                                ]

                            if original_query:
                                # Map feedback to action types
                                feedback_mapping = {
                                    "✅ This is correct": (True, None),
                                    "Should be SQL Query": (False, "SQL"),
                                    "Should be RAG Search": (False, "RAG"),
                                    "Should be Python Analysis": (False, "CODE"),
                                    "Should be Dataset Info": (False, "METADATA"),
                                }

                                was_correct, correct_action = feedback_mapping.get(
                                    feedback_option, (True, None)
                                )

                                try:
                                    # Send feedback to backend
                                    response = requests.post(
                                        "http://localhost:8000/feedback",
                                        json={
                                            "query": original_query,
                                            "action_taken": detected_type,
                                            "was_correct": was_correct,
                                            "correct_action": correct_action,
                                            "implicit_signals": {
                                                "timestamp": datetime.now().isoformat()
                                            },
                                        },
                                        timeout=5,
                                    )

                                    if response.status_code == 200:
                                        if was_correct:
                                            st.success(
                                                "✅ Thank you! Positive feedback recorded."
                                            )
                                        else:
                                            st.success(
                                                f"✅ Thank you! The system will learn that this should be {correct_action}."
                                            )
                                    else:
                                        st.error("Failed to record feedback")
                                except Exception as e:
                                    st.error(f"Could not connect to backend: {str(e)}")

                    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# Chat input
col_input, col_send = st.columns([5, 1])

with col_input:
    # Check for pending query from example button
    pending = st.session_state.pop("pending_query", None)
    user_input = st.text_input(
        "Message:",
        value=pending or "",
        placeholder="Ask me anything about the data...",
        label_visibility="collapsed",
        key="chat_input",
    )

with col_send:
    send_button = st.button("Send", type="primary", width="stretch")

# Process message
if send_button and user_input and user_input.strip():
    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )

    # Process with query engine
    with st.spinner("🤔 Thinking..."):
        try:
            result = st.session_state.engine.execute_query(user_input)

            # Extract response content
            if result.get("conversational_summary"):
                # Orchestrator mode with commentary - don't use summary as main message for code/viz
                if result.get("type") == "code":
                    response_text = "I've generated visualizations to help understand the data patterns."
                else:
                    response_text = result["conversational_summary"]
            elif result.get("natural_language_summary"):
                # Natural language summary mode
                response_text = result["natural_language_summary"]
            elif result.get("response"):
                # Conversation mode
                response_text = result["response"]
            elif result.get("answer"):
                # RAG mode
                response_text = result["answer"]
            elif result.get("message"):
                # Code generation mode
                response_text = result["message"]
            elif result.get("error"):
                response_text = f"❌ {result['error']}"
            else:
                response_text = "I processed your query. Check the data below!"

            # Process markdown formatting in response text
            import re

            response_text = re.sub(
                r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response_text
            )

            # Add assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "result": result,
                }
            )

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Footer
st.divider()
st.caption(
    "💬 CommServe Chat • Powered by Multi-Tier AI Architecture • Conversational Mode Active"
)
