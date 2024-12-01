import re

from openhands.agenthub.coact_agent.planner.action_parser import (
    CoActActionParserPhasePlan,
)
from openhands.agenthub.codeact_agent.action_parser import (
    CodeActActionParserAgentDelegate,
    CodeActActionParserCmdRun,
    CodeActActionParserFileEdit,
    CodeActActionParserFinish,
    CodeActActionParserIPythonRunCell,
    CodeActActionParserMessage,
    CodeActResponseParser,
)
from openhands.controller.action_parser import ActionParser
from openhands.events.action import (
    Action,
    AgentFinishAction,
)


class ExecutorResponseParser(CodeActResponseParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - IPythonRunCellAction(code) - IPython code to run
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    - AgentFinishAction() - end the interaction
    """

    def __init__(self):
        # Need pay attention to the item order in self.action_parsers
        super().__init__()
        self.action_parsers = [
            CodeActActionParserFinish(),
            CodeActActionParserFileEdit(),
            CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            CodeActActionParserAgentDelegate(),
            CoActActionParserRequest(),
            CoActActionParserPhaseTransition(),
        ]
        self.default_parser = CodeActActionParserMessage()

    # def parse_response(self, response) -> str:
    #     action = response.choices[0].message.content
    #     if action is None:
    #         return ''
    #     for action_suffix in [
    #         'bash',
    #         'ipython',
    #         'browse',
    #         'request',
    #     ]:
    #         if (
    #             f'<execute_{action_suffix}>' in action
    #             and f'</execute_{action_suffix}>' not in action
    #         ):
    #             action += f'</execute_{action_suffix}>'
    #     return action


class CoActActionParserRequest(ActionParser):
    def __init__(self):
        self.request = None

    def check_condition(self, action_str: str) -> bool:
        self.request = re.search(
            r'<execute_request>(.*)</execute_request>', action_str, re.DOTALL
        )
        return self.request is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.request is not None
        ), 'self.request should not be None when parse is called'

        replan_request = self.request.group(1).strip()
        return AgentFinishAction(
            thought=replan_request,
            outputs={'content': replan_request},
        )


class CoActActionParserPhaseTransition(ActionParser):
    def __init__(self):
        self.request = None
        # Initialize an instance of CoActActionParserPhasePlan
        self.phase_plan_parser = CoActActionParserPhasePlan()

    def check_condition(self, action_str: str) -> bool:
        self.request = re.search(
            r'<execute_global_plan>(.*)</execute_global_plan>',
            action_str,
            re.DOTALL,
        )
        return self.request is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.request is not None
        ), 'self.request should not be None when parse is called'

        phase_transition_request = self.request.group(1).strip()

        # Use the phase_plan_parser to parse the phase plan action
        if self.phase_plan_parser.check_condition(phase_transition_request):
            return self.phase_plan_parser.parse(phase_transition_request)
        else:
            # Handle cases where the phase plan doesn't match expected structure
            raise ValueError('Phase transition request format is invalid.')