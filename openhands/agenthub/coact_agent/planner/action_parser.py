import json
import re

from openhands.agenthub.codeact_agent.action_parser import (
    CodeActActionParserAgentDelegate,
    CodeActActionParserCmdRun,
    CodeActActionParserFinish,
    CodeActActionParserIPythonRunCell,
    CodeActActionParserMessage,
    CodeActResponseParser,
)
from openhands.controller.action_parser import ActionParser
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
)


class PlannerResponseParser(CodeActResponseParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - IPythonRunCellAction(code) - IPython code to run
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    - AgentFinishAction() - end the interaction
    """

    def __init__(self, initial_task_str=None):
        # Need pay attention to the item order in self.action_parsers
        super().__init__()
        self.action_parsers = [
            CodeActActionParserFinish(),
            CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            CodeActActionParserAgentDelegate(),
            CoActActionParserGlobalPlan(initial_task_str=initial_task_str),
        ]
        self.default_parser = CodeActActionParserMessage()

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        for action_suffix in [
            'bash',
            'ipython',
            'browse',
            'global_plan',
            'phase_plan',
            'decide',
            'revise',
            'overrule',
            'collation',
        ]:
            if (
                f'<execute_{action_suffix}>' in action
                and f'</execute_{action_suffix}>' not in action
            ):
                action += f'</execute_{action_suffix}>'
        return action


class CoActActionParserGlobalPlan(ActionParser):
    """Parser action:
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def __init__(
        self,
        initial_task_str: list | None = None,
    ):
        self.global_plan: re.Match | None = None
        self.initial_task_str = initial_task_str or ['']

    def check_condition(self, action_str: str) -> bool:
        self.global_plan = re.search(
            r'<execute_global_plan>(.*)</execute_global_plan>', action_str, re.DOTALL
        )
        return self.global_plan is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.global_plan is not None
        ), 'self.global_plan should not be None when parse is called'
        thought = action_str.replace(self.global_plan.group(0), '').strip()
        global_plan_actions = self.global_plan.group(1).strip()

        # Some extra processing when doing swe-bench eval: extract text up to and including '--- END ISSUE ---'
        issue_text_pattern = re.compile(r'(.*--- END ISSUE ---)', re.DOTALL)
        issue_text_match = issue_text_pattern.match(self.initial_task_str[0])

        if issue_text_match:
            self.initial_task_str[0] = issue_text_match.group(1)

        return AgentDelegateAction(
            agent='CoActExecutorAgent',
            thought=thought,
            inputs={
                'task': f'The user message is: {self.initial_task_str[0]}.\nExecute the following plan to fulfill it:\n{global_plan_actions}'
            },
            action_suffix='global_plan',
        )
        # 'task': f'The user message is: {self.initial_task_str[0]}.\nThis is the global plan:\n{global_plan_actions}\n\nYour task is to execute the following phase:\nPhase 1: {global_plan_json['Phase 1']}'


# give a class for executing next agent using the CoActActionParserGlobalPlan pattern
class CoActActionParserPhasePlan(ActionParser):
    """Parser action:
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def __init__(
        self,
        initial_task_str: list | None = None,
    ):
        self.phase_plan: re.Match | None = None
        self.initial_task_str = initial_task_str or ['']
        print(
            '\n\n--------------------------------------\nhey from init\n--------------------------------------\n\n'
        )

    def check_condition(self, action_str: str) -> bool:
        self.phase_plan = re.search(
            r'<phase_transition>(.*)</phase_transition>', action_str, re.DOTALL
        )
        return self.phase_plan is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.phase_plan is not None
        ), 'self.phase_plan should not be None when parse is called'
        thought = action_str.replace(self.phase_plan.group(0), '').strip()
        phase_plan_actions = self.phase_plan.group(1).strip()

        # Some extra processing when doing swe-bench eval: extract text up to and including '--- END ISSUE ---'
        issue_text_pattern = re.compile(r'(.*--- END ISSUE ---)', re.DOTALL)
        issue_text_match = issue_text_pattern.match(self.initial_task_str[0])

        if issue_text_match:
            self.initial_task_str[0] = issue_text_match.group(1)

        # save the phase plan in a dict, and see which is the phase to be executed next
        # if the phase is the last one, then return the agent finish action
        # else, return the agent delegate action
        # the agent delegate action will be used to execute the next phase
        phase_plan_json = json.loads(phase_plan_actions)
        phase_to_execute = None
        for phase in phase_plan_json:
            if phase['status'] == 'done':
                continue
            else:
                phase_to_execute = phase
                break

        if phase_to_execute is None:
            return AgentFinishAction(
                thought='',
                outputs={'content': ''},
            )

        return AgentDelegateAction(
            agent='CoActExecutorAgent',
            thought=thought,
            inputs={
                'task': f'The user message is: {self.initial_task_str[0]}.\nThis is the global plan:\n{phase_plan_actions}\n\nYour task is to execute the following phase:\n{phase_to_execute}: {phase_plan_json[phase_to_execute]}'
            },
            action_suffix='phase_plan',
        )
