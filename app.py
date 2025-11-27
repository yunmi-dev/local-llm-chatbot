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
    
    def __init__(self, model_name="gemma3:1b", temperature=0.7):
        """
        Args:
            model_name: 사용할 Ollama 모델명
            temperature: 응답의 창의성 조절 (0~1)
        """
        # Model 초기화
        self._model = ChatOllama(model=model_name, temperature=temperature)
        
        # Prompt Template 설정
        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요.

Question: {question}

Answer:"""
        
        self._prompt = ChatPromptTemplate.from_template(self._template)
        
        # Chain 연결 (LCEL - LangChain Expression Language)
        self._chain = (
            {'question': RunnablePassthrough()}
            | self._prompt
            | self._model
            | StrOutputParser()
        )
    
    def invoke(self, user_input: str) -> str:
        """
        사용자 입력을 받아 LLM 응답 반환
        
        Args:
            user_input: 사용자 질문
            
        Returns:
            LLM 응답 텍스트
        """
        try:
            response = self._chain.invoke(user_input)
            return response
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}\n\nOllama 서비스가 실행 중인지 확인해주세요."


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
            'primary': "#FBA1CE",          # 파스텔 핑크
            'secondary': "#FDADD5",        # 연한 핑크
            'accent': "#FC72B7",           # 진한 핑크
            
            # 배경 색상
            'background': '#FFF5FA',       # 아주 연한 핑크 배경
            'chat_bg': '#FFFFFF',          # 채팅 배경 (흰색)
            'sidebar_bg': '#FFE8F5',       # 사이드바 배경
            
            # 텍스트 색상
            'text_dark': '#4A4A4A',        # 진한 회색
            'text_light': '#8A8A8A',       # 연한 회색
            'text_white': '#FFFFFF',       # 흰색
            # 'text_white': "#C7538D", 
            
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
            .main {{
                background: linear-gradient(135deg, #2D1B2E 0%, #C7538D 100%) !important;
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            /* 메인 컨테이너 */
            .block-container {{
                padding-top: 3rem;
                padding-bottom: 3rem;
                max-width: 900px;
            }}
            
            /* 사이드바 스타일 - 폭 넓히기 */
            [data-testid="stSidebar"] {{
                background-color: {self.colors['sidebar_bg']};
                border-right: 1px solid {self.colors['secondary']};
                width: 350px !important;
                min-width: 350px !important;
            }}
            
            [data-testid="stSidebar"] > div:first-child {{
                width: 350px !important;
            }}
            
            [data-testid="stSidebar"] .element-container {{
                color: {self.colors['text_dark']};
            }}
            
            /* 헤더 스타일 */
            h1 {{
                color: #FFFFFF;
                font-weight: 600;
                font-size: 2rem;
                margin-bottom: 0.5rem;
                letter-spacing: -0.02em;
                text-align: center;
            }}
            
            /* 부제목 */
            .subtitle {{
                color: #E0E0E0;
                font-size: 0.95rem;
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
            
            /* 사용자 메시지 */
            [data-testid="stChatMessageContent"] {{
                background-color: transparent;
            }}
            
            /* 입력창 컨테이너 */
            .stChatInputContainer {{
                border-top: 1px solid {self.colors['secondary']};
                padding: 1.5rem 2rem;
                background: rgba(45, 27, 46, 0.95) !important;
                max-width: 100% !important;
            }}

            /* 입력창 래퍼 */
            [data-testid="stChatInput"] {{
                max-width: 100% !important;
                margin: 0 auto;
            }}

            /* 입력창 스타일 - 흰색 배경 */
            [data-testid="stChatInput"] > div {{
                background-color: #FFFFFF !important;
                border: 2px solid {self.colors['primary']} !important;
                border-radius: 28px !important;
                padding: 0 !important;
                width: 100% !important;
            }}

            /* 입력창 내부 */
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] input {{
                background-color: #FFFFFF !important;
                color: #2D1B2E !important;
                font-size: 1.1rem !important;
                padding: 19px 30px !important;
                min-height: 60px !important;
                border: none !important;
                border-radius: 28px !important;
            }}

            /* Placeholder 색상 */
            [data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInput"] input::placeholder {{
                color: #999999 !important;
                opacity: 1 !important;
            }}

            /* Focus 상태 */
            [data-testid="stChatInput"] > div:focus-within {{
                border-color: {self.colors['accent']} !important;
                box-shadow: 0 0 0 3px rgba(255, 179, 217, 0.3) !important;
            }}

            /* 전송 버튼 */
            [data-testid="stChatInput"] button {{
                background-color: {self.colors['primary']} !important;
                color: #2D1B2E !important;
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
                color: {self.colors['text_white']};
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
            
            /* Expander 스타일 - 배경 진하게 + 글자색 */
            .streamlit-expanderHeader {{
                background-color: rgba(255, 179, 217, 0.25) !important;
                border-radius: 8px;
                color: #2D1B2E !important;
                font-weight: 600 !important;
                padding: 0.75rem 1rem !important;
                border: 1.5px solid {self.colors['primary']};
            }}
            
            .streamlit-expanderHeader:hover {{
                background-color: rgba(255, 179, 217, 0.4) !important;
            }}
            
            /* Expander 펼쳤을 때 */
            details[open] > summary {{
                background-color: rgba(255, 158, 206, 0.4) !important;
                border-bottom: 2px solid {self.colors['primary']};
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
            st.markdown("### 색상 테마")
            st.markdown(f"""
                <div style="display: flex; gap: 8px; margin-bottom: 1rem;">
                    <div style="width: 30px; height: 30px; background-color: {self.colors['primary']}; 
                         border-radius: 50%; border: 2px solid white;"></div>
                    <div style="width: 30px; height: 30px; background-color: {self.colors['secondary']}; 
                         border-radius: 50%; border: 2px solid white;"></div>
                    <div style="width: 30px; height: 30px; background-color: {self.colors['accent']}; 
                         border-radius: 50%; border: 2px solid white;"></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 모델 정보 표기
            with st.expander("모델 정보", expanded=True):
                st.markdown("""
                **모델**: gemma3:1b  
                **제공**: Google  
                **타입**: 로컬 LLM  
                **프레임워크**: LangChain
                """)
            
            # 통계
            if "messages" in st.session_state:
                user_msg_count = len([m for m in st.session_state["messages"] if m.role == "user"])
                with st.expander("대화 통계"):
                    st.metric("💬 질문 수", user_msg_count)
            
            # 프로젝트 정보
            with st.expander("프로젝트 정보"):
                st.markdown("""
                **과목**: 모바일/웹서비스프로그래밍  
                **학교**: 경희대학교  
                **개발자**: 정윤미
                """)
            
            st.markdown("---")
            
            # 초기화 버튼
            if st.button("🗑️ 대화 초기화 버튼"):
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
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("생각하는 중..."):
                    msg_assistant = self._llm.invoke(user_input)
                st.write(msg_assistant)
                st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=msg_assistant)
                )


def main():
    """메인 실행 함수"""
    # LLM 초기화
    llm = ChatLLM(model_name="gemma3:1b", temperature=0.7)
    
    # 웹 인터페이스 초기화 및 실행
    web = ChatWeb(
        llm=llm,
        page_title="Chatbot Service",
        page_icon="💬"
    )
    web.run()


if __name__ == '__main__':
    main()