import os
import re
import logging
from openai import OpenAI

logger = logging.getLogger('RRAG-main')

# Location file for the shared gpt-oss-120b vLLM server on Wulver (see
# CONNECT_TO_SHARED_VLLM.md). Kept in sync with src/models.py's default.
SHARED_VLLM_SERVER_FILE = os.environ.get(
    "SHARED_VLLM_SERVER_FILE", "/project/ss797/ap2645/vllm_server.txt"
)

judge_prompt = '''Here is the complete answer:

<Start of Answer>
{answer}
<End of Answer>

an LLM provides to the following question:

<Start of Question>
{question}
<End of Question>

Extract the ultimate answer provided by the LLM to the question, without any additional analysis, thinking, internal notes, etc. Provide your answer in the following format:

<Answer>
[Your Answer].
</Answer>

Example: Given the following complete answer:

<Start of Answer>
The first document provides irrelevant information to the question. The second document says Jack wins the prize at 2011 but seems incorrect.
The third document says Jack wins the prize at 2005 and seems more trustworthy. Thus, my answer is 2005.
<End of Answer>

an LLM provides to the following question:

<Start of Question>
When did Jack win the Nobel Prize?
<End of Question>

The answer you should provide is the following:

<ANSWER>
2005.
</ANSWER>

If the answer provided by the LLM is in a multiple-choice format, include the choice as well, e.g.

<Start of Answer>
... Thus, my answer is A. 2005.
<End of Answer>

<ANSWER>
A. 2005.
</ANSWER>
'''

class LLMJudge(object):
    """Post-processes astuterag/instructrag_icl responses to extract the final answer.

    By default this talks to the shared gpt-oss-120b vLLM server on Wulver
    (no API key needed). Set JUDGE_BACKEND=openai (and OPENAI_API_KEY) to use
    a real OpenAI model instead, e.g. when not running on Wulver.
    """
    def __init__(self):
        self.backend = os.environ.get("JUDGE_BACKEND", "shared_vllm")
        if self.backend == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            self.model = "gpt-4o"
            self.max_tokens = 1000
        else:
            self.server_file = os.environ.get("SHARED_VLLM_SERVER_FILE", SHARED_VLLM_SERVER_FILE)
            self.model = "openai/gpt-oss-120b"
            # gpt-oss is a reasoning model: its hidden reasoning shares the
            # max_tokens budget, so this needs more headroom than gpt-4o did.
            self.max_tokens = 2048

    def _client(self):
        if self.backend == "openai":
            return self.client
        with open(self.server_file, 'r') as f:
            node_port = f.read().strip()
        return OpenAI(base_url=f"http://{node_port}/v1", api_key="dummy")

    def judge(self, question, answer):
        final_prompt = judge_prompt.format(question=question, answer=answer)
        response = self.get_gpt_output(self.model, final_prompt)
        final_response = self.extract_from_text(response, "ANSWER")
        logger.debug(f"Final response after post-processing by LLM judge: {final_response}")
        return final_response

    def get_gpt_output(self, model, prompt, temperature=0):
        messages = [{"role": "user", "content": prompt}]
        try:
            client = self._client()
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=self.max_tokens,
                top_p=0.5,
                messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"LLM Judge error getting output: {e}")
            return ""

    def extract_from_text(self, text, tag):
        # extract stuffs inside <tag>...</tag>
        try:
            pattern = fr'<{tag}>\s*(.*?)\s*</{tag}>'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception as e:
            logger.warning("LLM Judge error extracting from text:", e)
            return ""