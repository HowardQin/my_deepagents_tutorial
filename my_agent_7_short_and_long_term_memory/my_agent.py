import os
from pathlib import Path
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends import StoreBackend, StateBackend, CompositeBackend
from dotenv import load_dotenv
import requests
from langchain_core.tools import tool
from openai import OpenAI

load_dotenv()

# 这些环境变量可以写在.env中
agent_model = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0,
)

# deepagent 需要一个文件系统存放中间结果，
# 这里设置一个虚拟文件系统，
# 当前目录下的 fs 作为虚拟文件系统的根目录“/”

CURRENT_DIR = Path(__file__).parent.resolve()
FS_ROOT = CURRENT_DIR / "fs"
my_backend = LocalShellBackend(root_dir=str(FS_ROOT), virtual_mode=True)
composite_backend = lambda rt: CompositeBackend(
    default=my_backend,
    routes={
        "/path1": StoreBackend(rt),
        "/path2/": StateBackend(rt),
        "/path3/": LocalShellBackend(root_dir=str(FS_ROOT / "path3"), virtual_mode=True)
    }
)

@tool(parse_docstring=True)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    Args:
        to: target email address.
        subject: topic of the email body.
        body: content of the email.

    Returns:
        Message of sending email.
    """
    body
    return f"Sent email to {to} of subject {subject}."

@tool(parse_docstring=True)
def image_gen_tool(input_desc: str, ouput_dir: str, file_name: str) -> str:
    """Tool for generating image from text, image file type is png

    Args:
        input_desc: Description text of the image to be generated.
        ouput_dir: The directory in virtual filesystem where the generated image saved, start with '/'.
        file_name: file name of generated image without suffix.

    Returns:
        Message of image generation result and where the image saved.
    """
    client = OpenAI(api_key=os.getenv("IMAGE_API_KEY"), 
                    base_url=os.getenv("IMAGE_BASE_URL"))

    response = client.images.generate(
        model=os.getenv("IMAGE_GEN_MODEL_NAME"),
        prompt=input_desc,
        size="1024x1024",
        n=2,
        extra_body={
            "step": 20
        }
    )
    # 硅基流动生成图片后，会返回图片url，提取这个url，然后下载图片
    url = response.images[0]['url']
    # 下载图片
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # 需要将下载的图片（response.content）上传到虚拟文件系统的目录下
        my_backend.upload_files([(f"{ouput_dir}/{file_name}.png", response.content)])
        return f"Successfully generated and saved image to {ouput_dir}/{file_name}.png"
    else:
        return f"Failed to generate or save image. Status code: {response.status_code}"


# 子智能体提示词
IMG_GEN_INSTRUCTIONS = """ 
You are a excellent image creator, 
use input text (add more concrete descriptions if necessary) to generate images.
"""

# 将 image_gen_tool 工具给子智能体，子智能体使用这个工具生成图片，
# 生成图片过程产生的上下文，不会出现在父智能体的上下文中
# 子智能体也有middleware、backend等配置，如果不配置，默认继承父智能体
image_gen_agent = {
    "name": "image-gen-agent",
    "description": "根据文字描述生成图像",
    "system_prompt": IMG_GEN_INSTRUCTIONS,
    "tools": [image_gen_tool],
    "interrupt_on": {"image_gen_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
}

SYSTEM_PROMPT="""
    在 /path1/character.md 中记录你的性格。
    在 /path2/user.md 中记录用户偏好和个人信息。
    在 /path3/background.md 中记录背景知识。
    当用户在对话中提及个人信息、偏好或设置智能体的性格时，
    使用edit_file工具，将这些信息更新到相应的文件中。
"""

agent = create_deep_agent(
    model=agent_model,
    subagents=[image_gen_agent],
    backend=composite_backend,
    system_prompt=SYSTEM_PROMPT,
    skills=["/skills"],
    debug=True,
    tools=[send_email],
    interrupt_on={"send_email":True}
)

def main():
    pass

if __name__ == "__main__":
    main()
