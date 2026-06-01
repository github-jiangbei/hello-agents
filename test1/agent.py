import os
import re
import requests
import httpx
from openai import OpenAI

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对 Thought-Action
- Action 必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
"""


def get_weather(city: str) -> str:
    """通过 wttr.in 查询天气。"""
    url = f"https://wttr.in/{city}?format=j1"

    try:
        session = requests.Session()
        # 忽略系统环境里的错误代理；如果你必须使用代理，可改成 True
        session.trust_env = False

        response = session.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}当前天气：{weather_desc}，气温 {temp_c} 摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气搜索景点。
    这里不用 tavily-python SDK，直接 HTTP 调用，避免 TavilyClient 导入版本问题。
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量。"

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": f"{city} 在 {weather} 天气下最值得去的旅游景点推荐及理由",
        "search_depth": "basic",
        "include_answer": True,
        "max_results": 5,
    }

    try:
        session = requests.Session()
        # 忽略系统环境里的错误代理；如果你必须使用代理，可改成 True
        session.trust_env = False

        response = session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("answer"):
            return data["answer"]

        results = data.get("results", [])
        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(
            f"- {item.get('title', '')}: {item.get('content', '')}"
            for item in results
        )

    except Exception as e:
        # 为了先跑通 1.3.2/1.3.3，这里给一个可继续循环的兜底结果
        return (
            f"搜索接口暂时不可用：{e}\n"
            f"演示推荐：如果{city}天气为 {weather}，可以优先考虑颐和园、天坛公园或北海公园。"
        )


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


class OpenAICompatibleClient:
    """调用兼容 OpenAI 接口的大语言模型服务。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            # 忽略系统环境里的错误代理；如果你必须使用代理，可改成 trust_env=True
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大语言模型...")

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
            )

            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer

        except Exception as e:
            print(f"调用 LLM API 时发生错误：{e}")
            return "错误：调用语言模型服务时出错。"


def parse_action(llm_output: str):
    """从模型输出里解析 Action。"""
    action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)
    if not action_match:
        return None, None, "未能解析到 Action 字段。"

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        if not finish_match:
            return None, None, "Finish 格式不正确，应该是 Finish[最终答案]。"
        return "Finish", {"answer": finish_match.group(1).strip()}, None

    tool_name_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str)

    if not tool_name_match or not args_match:
        return None, None, f"无法解析工具调用格式：{action_str}"

    tool_name = tool_name_match.group(1)
    args_str = args_match.group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    return tool_name, kwargs, None


def run_agent(user_prompt: str, max_steps: int = 5) -> str:
    """智能体主循环。"""
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model_id = os.environ.get("LLM_MODEL_ID")

    if not api_key:
        return "错误：未配置 LLM_API_KEY 环境变量。"
    if not base_url:
        return "错误：未配置 LLM_BASE_URL 环境变量。"
    if not model_id:
        return "错误：未配置 LLM_MODEL_ID 环境变量。"

    llm = OpenAICompatibleClient(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )

    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}")
    print("=" * 40)

    for step in range(max_steps):
        print(f"--- 循环 {step + 1} ---\n")

        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        # 防止模型一次输出多组 Thought-Action
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL,
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对。")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        tool_name, kwargs, error = parse_action(llm_output)

        if error:
            observation = f"错误：{error}"
            observation_str = f"Observation: {observation}"
            print(observation_str)
            print("=" * 40)
            prompt_history.append(observation_str)
            continue

        if tool_name == "Finish":
            return kwargs["answer"]

        if tool_name not in available_tools:
            observation = f"错误：未定义的工具 '{tool_name}'"
        else:
            try:
                observation = available_tools[tool_name](**kwargs)
            except Exception as e:
                observation = f"错误：工具 {tool_name} 执行失败 - {e}"

        observation_str = f"Observation: {observation}"
        print(observation_str)
        print("=" * 40)
        prompt_history.append(observation_str)

    return "错误：达到最大循环次数，任务仍未完成。"


if __name__ == "__main__":
    user_input = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
    final_answer = run_agent(user_input)

    print("\n最终输出：")
    print(final_answer)
