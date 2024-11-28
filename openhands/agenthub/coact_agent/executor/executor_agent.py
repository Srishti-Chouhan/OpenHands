import os

from openhands.agenthub.coact_agent.executor.action_parser import ExecutorResponseParser
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig
from openhands.llm.llm import LLM
from openhands.runtime.plugins.agent_skills import AgentSkillsRequirement
from openhands.utils.prompt import PromptManager


class LocalExecutorAgent(CodeActAgent):
    VERSION = '1.0'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        # llm.config.model = 'anthropic/neulab/claude-3-5-sonnet-20241022'
        super().__init__(llm, config)

        # self.function_calling_active = False

        self.action_parser = ExecutorResponseParser()
        self.prompt_manager = PromptManager(
            prompt_dir=os.path.join(os.path.dirname(__file__)),
            agent_skills_docs=AgentSkillsRequirement.documentation,
            micro_agent=self.micro_agent,
        )

        self.system_prompt = self.prompt_manager.system_message
        self.initial_user_message = self.prompt_manager.initial_user_message

        # self.messages.append(
        #         Message(
        #             role='user',
        #             content=[TextContent(text='The user task is: build a simple calculator app. Execute the global plan given by the planner agent to fulfill the user task')],
        #         )
        #     )

        self.params['stop'].append('</execute_request>')
