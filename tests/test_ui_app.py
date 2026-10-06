from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rag_law"
    / "ui"
    / "app.py"
)


def test_streamlit_landing_page_renders() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=15)

    assert not app.exception
    assert app.selectbox[0].value == "全部法律"
    assert app.text_input[0].placeholder == "描述你的法律问题…"
    assert {button.label for button in app.button} == {
        "清空当前对话",
        "公司一直没和我签书面劳动合同，我可以要求什么？",
        "酒后驾驶机动车会受到什么处罚？",
        "消费者买到不符合食品安全标准的食品，可以要求赔偿吗？",
        "查找相关法条",
    }
