from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
import dataclasses
from dataclasses import dataclass, field

@dataclasses.dataclass
class Conversation:
    # simplier and more flexible
    system_template: str = "{system_message}\n\n"
    utterance_template: str = '{role}: {message}\n\n'
    query_template: str = '{role}: '
    #
    system_message: str = ""
    roles: Dict[str, str] = field(default_factory=lambda: {"user": "USER", "assistant": "ASSISTANT"})
    # All messages. Each item is (role, message).
    # equal to fschat `messages`
    utterances: List[str] = ()
    # The number of few shot examples + 1
    # because of the system prompt
    offset: int = 0

    def get_prompt(self) -> str:
        ret = self.system_template.format(system_message=self.system_message)
        for utter in self.utterances:
            ret += utter
        return ret

    def append_message(self, role, message, is_query=False):
        if is_query:
            self.utterances.append(self.query_template.format(role=role, message=message))
        else:
            self.utterances.append(self.utterance_template.format(role=role, message=message))

    def from_openai(self, request):
        for message in request.messages:
            msg_role = message["role"]
            # TODO: 规定如果system_message不存在 或者system_message为空，则使用默认的system prompt
            if msg_role == "system":
                if message["content"] != '':
                    self.system_message = message["content"]
            elif msg_role in ["user", "assistant"]:
                self.append_message(self.roles[msg_role], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        self.append_message(self.roles['assistant'], '', is_query=True)

    def to_gradio(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.utterances[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.utterances[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret