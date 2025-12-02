"""
ChatBot with Streamlit & Ollama
Kyung Hee University - Web Service Programming Assignment
"""

import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage


class ChatLLM:
    """Ollama LLM을 이용한 챗봇 로직 클래스"""
    
    def __init__(self, model_name="gemma2:2b", temperature=0.7):
        """
        Args:
            model_name: 사용할 Ollama 모델명
            temperature: 응답의 창의성 조절 (0~1)
        """
        # Model 초기화
        self._model = ChatOllama(model=model_name, temperature=temperature)
    
    def invoke(self, user_input: str) -> str:
        """
        사용자 입력을 받아 LLM 응답 반환
        
        Args:
            user_input: 사용자 질문
            
        Returns:
            LLM 응답 텍스트
        """
        try:
            # 대화 기록 가져오기
            messages = []

            if "messages" in st.session_state:
                # 이전 대화를 LangChain 형식으로 변환
                for msg in st.session_state["messages"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            # 현재 사용자 입력 추가
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # LLM 호출 (대화 기록 포함)
            response = self._model.invoke(messages)
            return response.content
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}\n\nOllama 서비스가 실행 중인지 확인해주세요."
    
    def stream(self, user_input: str):
        """
        스트리밍 방식으로 응답 생성
        """
        try:
            messages = []
            if "messages" in st.session_state:
                for msg in st.session_state["messages"]:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            
            messages.append({
                "role": "user",
                "content": user_input
            })
            
            # 스트리밍 호출
            for chunk in self._model.stream(messages):
                yield chunk.content
                
        except Exception as e:
            yield f"오류가 발생했습니다: {str(e)}\n\nOllama 서비스가 실행 중인지 확인해주세요."


class ChatWeb:
    """Streamlit 웹 인터페이스 클래스"""
    
    def __init__(self, llm: ChatLLM, page_title="Chatbot Service", page_icon="💬"):
        """
        Args:
            llm: ChatLLM 인스턴스
            page_title: 웹페이지 제목
            page_icon: 웹페이지 아이콘
        """
        self._llm = llm
        self._page_title = page_title
        self._page_icon = page_icon

        self.colors = {
            # 메인 색상
            'primary': "#EA7DB3",
            'secondary': "#FFC9E4",
            'accent': "#FF6EB6",
            'aaaccent': "#E16AA6",
            
            # 배경 색상
            'background': '#FFF5FA',       # 배경
            'chat_bg': '#FFFFFF',          # 채팅 배경
            'sidebar_bg': "#FEC5E2",       # 사이드바 배경
            
            # 텍스트 색상
            'text_dark': '#4A4A4A',
            'text_light': '#8A8A8A',
            'text_white': '#FFFFFF',
            
            # 메시지 색상
            'user_msg': '#FFE8F5',         # 사용자 메시지 배경
            'assistant_msg': '#F5F5F5',    # AI 메시지 배경
        }
    
    def print_messages(self):
        """세션에 저장된 이전 대화 기록 출력"""
        if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
            for chat_message in st.session_state["messages"]:
                with st.chat_message(chat_message.role):
                    st.markdown(chat_message.content)
    
    def run(self):
        """Streamlit 앱 실행"""
        # 웹 페이지 기본 설정
        st.set_page_config(
            page_title=self._page_title,
            page_icon=self._page_icon,
            layout="centered",
            initial_sidebar_state="expanded"
        )

        # CSS
        st.markdown(f"""
            <style>
            /* 전체 배경색 */
            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            .stApp {{
                background: linear-gradient(135deg, #FFE8F5 0%, #FFF5FA 100%) !important;
            }}

            .main {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            /* 메인 컨테이너 */
            .block-container {{
                padding-top: 3rem;
                padding-bottom: 3rem;
                max-width: 900px;
            }}
            
            /* 사이드바 스타일 */
            [data-testid="stSidebar"] {{
                background-color: {self.colors['sidebar_bg']} !important;
                border-right: 1px solid {self.colors['secondary']};
            }}

            [data-testid="stSidebar"] > div:first-child {{
                background-color: {self.colors['sidebar_bg']} !important;
            }}
            
            [data-testid="stSidebar"] .element-container {{
                color: {self.colors['text_dark']};
            }}
            
            /* 헤더 스타일 */
            h1 {{
                color: #E16AA6 !important;
                font-weight: 600;
                font-size: 2rem;
                margin-bottom: 0.5rem;
                margin-right: 20rem;
                letter-spacing: -0.02em;
                text-align: center;
            }}
            
            /* 부제목 */
            .subtitle {{
                color: #E16AA6 !important;
                font-size: 2rem;
                font-weight: 400;
                margin-bottom: 2rem;
                text-align: center;
            }}

            /* 채팅 메시지 컨테이너 */
            .stChatMessage {{
                background-color: {self.colors['chat_bg']};
                border-radius: 16px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(255, 179, 217, 0.15);
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }}

            /* 채팅 메시지 내부 모든 텍스트 색상 고정 */
            .stChatMessage p,
            .stChatMessage span,
            .stChatMessage strong,
            .stChatMessage em,
            .stChatMessage code,
            .stChatMessage li,
            .stChatMessage div {{
                color: {self.colors['text_dark']} !important;
            }}

            /* 인라인 코드 블록 */
            .stChatMessage code {{
                background-color: rgba(234, 125, 179, 0.1) !important;
                color: {self.colors['text_dark']} !important;
                padding: 2px 6px !important;
                border-radius: 4px !important;
            }}

            /* 코드 블록 */
            .stChatMessage pre {{
                background-color: rgba(234, 125, 179, 0.1) !important;
                border: 1px solid {self.colors['secondary']} !important;
                border-radius: 8px !important;
                padding: 12px !important;
            }}

            .stChatMessage pre code {{
                color: {self.colors['text_dark']} !important;
            }}
            
            /* 사용자 메시지 */
            [data-testid="stChatMessageContent"] {{
                background-color: transparent;
            }}

            /* 하단 영역 배경색 통일 */
            [data-testid="stHeader"] {{
                background-color: transparent !important;
            }}

            [data-testid="stBottom"],
            [data-testid="stBottom"] > *,
            footer,
            footer > * {{
                background: linear-gradient(135deg, #FFE8F5 0%, #FFF5FA 100%) !important;
            }}
            
            /* 입력창 컨테이너 */
            .stChatInputContainer {{
                border-top: 1px solid {self.colors['secondary']};
                padding: 1.5rem 2rem;
                background: rgba(45, 27, 46, 0.95) !important;
                max-width: 100% !important;
            }}

            /* 모든 하단 요소 배경 제거 */
            section[data-testid="stBottom"],
            section[data-testid="stBottom"] > *,
            section[data-testid="stBottom"] * {{
                background-color: #FFF5FA !important;
            }}

            /* 입력창 래퍼 */
            [data-testid="stChatInput"] {{
                max-width: 100% !important;
                margin: 0 auto;
            }}

            /* 입력창 스타일 */
            [data-testid="stChatInput"] > div {{
                background-color: #FFFFFF !important;
                border: 2px solid {self.colors['primary']} !important;
                border-radius: 28px !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                width: 100% !important;
                min-height: 65px !important;
                height: auto !important;
            }}

            /* 입력창 내부 */
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] input {{
                background-color: #FFFFFF !important;
                color: #2D1B2E !important;
                font-size: 1.1rem !important;
                padding: 19px 30px !important;
                min-height: 60px !important;
                height: 60px !important;
                max-height: 200px !important;
                border: none !important;
                border-radius: 28px !important;
                flex: 1 !important;
                width: 100% !important;
                line-height: 1.5 !important;
                resize: vertical !important;
            }}

            /* 입력창 최외곽 컨테이너 고정 */
            section[data-testid="stBottom"] {{
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                width: 100% !important;
                max-width: 100vw !important;
                background: linear-gradient(135deg, #FFE8F5 0%, #FFF5FA 100%) !important;
            }}

            /* 입력창 wrapper 너비 고정 */
            [data-testid="stChatInput"] {{
                width: 100% !important;
                max-width: 100% !important;
                margin: 0 auto !important;
            }}

            [data-testid="stChatInput"] > div {{
                width: 100% !important;
                max-width: none !important;
            }}

            /* Placeholder 색상 */
            [data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInput"] input::placeholder {{
                color: #999999 !important;
            }}

            /* Focus 상태 */
            [data-testid="stChatInput"] > div:focus-within {{
                border-color: {self.colors['accent']} !important;
                box-shadow: 0 0 0 3px rgba(255, 179, 217, 0.3) !important;
            }}

            /* 전송 버튼 */
            [data-testid="stChatInput"] button {{
                background-color: #E16AA6 !important;
                color: #FFFFFF !important;
                border: none !important;
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                padding: 0 !important;
                margin-right: 18px !important;
                align-self: center !important;
                position: relative !important;
                top: 0 !important;
            }}

            /* 전송 버튼 내부 아이콘 */
            [data-testid="stChatInput"] button svg {{
                margin-left: 4px !important;
            }}

            [data-testid="stChatInput"] button:hover {{
                background-color: {self.colors['accent']} !important;
            }}
            
            /* 버튼 스타일 */
            .stButton > button {{
                background-color: {self.colors['primary']};
                color: #FFFFFF !important;
                border: none;
                border-radius: 10px;
                padding: 0.6rem 1.5rem;
                font-weight: 500;
                transition: all 0.2s ease;
                width: 100%;
            }}

            .stButton > button:hover {{
                background-color: {self.colors['accent']};
                box-shadow: 0 4px 12px rgba(255, 158, 206, 0.4);
                transform: translateY(-2px);
            }}

            .stButton > button,
            .stButton > button p,
            .stButton > button span,
            .stButton button[kind="primary"],
            .stButton button[kind="secondary"] {{
                color: #FFFFFF !important;
            }}

            .stButton > button * {{
                color: #FFFFFF !important;
            }}
            
            /* Expander 스타일*/
            .streamlit-expanderHeader {{
                background-color: rgba(255, 158, 206, 0.4) !important;
                border-radius: 8px;
                color: #FFFFFF !important;
                font-weight: 600 !important;
                padding: 0.75rem 1rem !important;
                border: 1.5px solid {self.colors['primary']};
            }}
            
            .streamlit-expanderHeader:hover {{
                background-color: rgba(255, 179, 217, 0.5) !important;
            }}

            /* Expander 펼쳤을 때 */
            details[open] > summary {{
                background-color: {self.colors['primary']} !important;
                border-bottom: 2px solid {self.colors['primary']};
                border-radius: 8px 8px 0 0;
            }}

            details:not([open]) > summary {{
                background-color: {self.colors['primary']} !important;
                border-radius: 8px;
            }}
            
            /* Expander 내용 배경 */
            .streamlit-expanderContent {{
                background-color: rgba(255, 232, 245, 0.3) !important;
                border: 1px solid {self.colors['secondary']};
                border-top: none;
                border-radius: 0 0 8px 8px;
                padding: 0.5rem 1rem !important;
            }}
            
            /* 구분선 */
            hr {{
                border: none;
                border-top: 1px solid {self.colors['secondary']};
                margin: 1.5rem 0;
                opacity: 0.6;
            }}
            
            /* 메트릭 스타일 */
            [data-testid="stMetricValue"] {{
                color: {self.colors['primary']};
                font-weight: 600;
            }}
            
            /* 스피너 */
            .stSpinner > div {{
                border-top-color: {self.colors['primary']} !important;
            }}
            
            /* 스크롤바 */
            ::-webkit-scrollbar {{
                width: 10px;
                height: 10px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: rgba(45, 27, 46, 0.5);
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {self.colors['primary']};
                border-radius: 5px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {self.colors['accent']};
            }}
            
            /* Info box */
            .element-container div[data-testid="stMarkdownContainer"] p {{
                color: {self.colors['text_dark']};
            }}
            
            /* 사이드바 헤더 */
            [data-testid="stSidebar"] h2 {{
                color: {self.colors['text_dark']};
                font-weight: 600;
                font-size: 1.2rem;
            }}
            
            [data-testid="stSidebar"] h3 {{
                color: {self.colors['text_dark']};
                font-weight: 500;
                font-size: 1rem;
            }}
            
            /* 채팅 영역 중앙 정렬 */
            [data-testid="stChatMessageContainer"] {{
                max-width: 850px;
                margin: 0 auto;
            }}
            </style>
        """, unsafe_allow_html=True)
        
        # 헤더
        st.markdown(f"# {self._page_icon} {self._page_title}")
        st.markdown(
            f'<p class="subtitle">경희대학교 웹서비스프로그래밍 · Powered by Ollama</p>', 
            unsafe_allow_html=True
        )
        
        # 사이드바
        with st.sidebar:
            # 모델 설정
            with st.expander("모델 설정", expanded=False):
                temperature = st.slider(
                    "응답 창의성",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="값이 높을수록 더 창의적인 답변"
                )
                if temperature != 0.7:
                    self._llm._model.temperature = temperature
            
            # 대화 내보내기
            with st.expander("대화 저장", expanded=False):
                if st.button("대화 내용 다운로드"):
                    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
                        chat_text = ""
                        for msg in st.session_state["messages"]:
                            role = "사용자" if msg.role == "user" else "AI"
                            chat_text += f"{role}: {msg.content}\n\n"
                        
                        st.download_button(
                            label="📥 TXT 다운로드",
                            data=chat_text,
                            file_name="chat_history.txt",
                            mime="text/plain"
                        )
            
            # 모델 정보
            with st.expander("모델 정보", expanded=False):
                st.markdown("""
                **모델**: gemma2:2b  
                **제공**: Google  
                **타입**: 로컬 LLM  
                **프레임워크**: LangChain
                """)
            
            # 프로젝트 정보
            with st.expander("프로젝트 정보", expanded=False):
                st.markdown("""
                **과목**: 모바일/웹서비스프로그래밍  
                **학교**: 경희대학교  
                **개발자**: 정윤미
                """)
            
            st.markdown("---")
            
            # 초기화 버튼
            if st.button("대화 초기화"):
                st.session_state["messages"] = []
                st.rerun()
        
        # 대화 기록 초기화
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            # 환영 메시지
            welcome_msg = ChatMessage(
                role="assistant", 
                content="안녕하세요!👋 무엇을 도와드릴까요?"
            )
            st.session_state["messages"].append(welcome_msg)
        
        # 이전 대화 출력
        self.print_messages()
                
        # 사용자 입력
        if user_input := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 추가
            st.chat_message("user").write(user_input)
            st.session_state["messages"].append(
                ChatMessage(role="user", content=user_input)
            )
            
            # AI 응답 생성 (스트리밍)
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in self._llm.stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=full_response)
                )


def main():
    """메인 실행 함수"""
    # LLM 초기화
    llm = ChatLLM(model_name="gemma2:2b", temperature=0.7)
    
    # 웹 인터페이스 초기화 및 실행
    web = ChatWeb(
        llm=llm,
        page_title="Chatbot Service",
        page_icon="💬"
    )
    web.run()


if __name__ == '__main__':
    main()