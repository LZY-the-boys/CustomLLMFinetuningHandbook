# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import gradio as gr
import mdtex2html

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def main(
    *, 
    model: str='Qwen/Qwen-14B', 
    share: bool=False,
    inbrowser: bool=False,
    server_port: int=8080,
    server_name: str="127.0.0.1",
):

    from client.openai_client import chat_compeletion_openai_stream, LOG

    def predict(_query, _chatbot, _task_history):

        LOG.info(f"History: {_task_history}")
        LOG.info(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""

        messages = []
        for his in _task_history:
            messages.append({'role': 'user', 'content': his[0]})
            messages.append({'role': 'assistant', 'content': his[1]})
        messages.append({'role': 'user', 'content': _query})

        for response in chat_compeletion_openai_stream(
            model,
            messages,
        ):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))
            yield _chatbot
            full_response = _parse_text(response)

        _task_history.append((_query, full_response))
        LOG.info(f"CCIIP-GPT: {_parse_text(full_response)}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""<center><font size=8>CCIIP LAB ChatBOT</center>""")

        chatbot = gr.Chatbot(label='CCIIP-GPT', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

    demo.queue().launch(
        share=share,
        inbrowser=inbrowser,
        server_port=server_port,
        server_name=server_name,
    )


if __name__ == '__main__':
    # debug : use jurigged (auto reload)
    import defopt
    defopt.run(main)