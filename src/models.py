import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList
from .helper import StopOnTokens
#import deepspeed


from torch import LongTensor, FloatTensor

import logging
logger = logging.getLogger('RRAG-main')

from .reasoning_log import log_reasoning, usage_dict
from . import json_trace_log

import litellm
from litellm import batch_completion
from litellm.exceptions import RateLimitError
from openai import OpenAI
import os 
import json
import time
import joblib
from .prompt_template import *

# for question-answering
MAX_NEW_TOKENS = 50

# for biogen
# MAX_NEW_TOKENS = 2048
CONTEXT_MAX_TOKENS = {'Mistral-7B-Instruct-v0.2': 8192,
                      'Llama-3.2-3B-Instruct': 8192,
                      'Llama-3.2-1B-Instruct': 4096,
                      'Llama-3.1-8B-Instruct': 8192,
                      'Mixtral-8x7B-Instruct-v0.1': 32000,
                      'DeepSeek-R1-Distill-Qwen-7B': 8192,
                      'gpt-4o':8192,
                      'o1-mini':8192,
                      'vicuna-7b-v1.5': 4096,
                      'openai/gpt-oss-120b': 32768,
                      'Qwen/Qwen2.5-32B-Instruct': 32768}

# Location file for the shared gpt-oss-120b vLLM server on Wulver (see
# CONNECT_TO_SHARED_VLLM.md). The node:port changes whenever the server is
# restarted, so we always read this file fresh rather than caching the value.
SHARED_VLLM_SERVER_FILE = os.environ.get(
    "SHARED_VLLM_SERVER_FILE", "/project/ss797/ap2645/vllm_server.txt"
)

# Separate location file for the shared Qwen2.5-32B-Instruct vLLM server
# (a different server, owned by a different lab member). Same rules: read
# fresh on every query since the node:port changes on restart.
QWEN_VLLM_SERVER_FILE = os.environ.get(
    "QWEN_VLLM_SERVER_FILE", "/project/ss797/cs757/KidnapRAG/vllm_server.txt"
)

# Address file published by scripts/serve_gpt_oss_local.slurm for a gpt-oss-120b
# server you run yourself from locally-downloaded weights (your own job, so you
# control concurrency). Default lives in $HOME so it needs no username.
LOCAL_VLLM_SERVER_FILE = os.environ.get(
    "LOCAL_VLLM_SERVER_FILE", os.path.expanduser("~/vllm_local_server.txt")
)

# Groq's hosted gpt-oss-120b, for testing on your own machine (no Wulver
# access needed). OpenAI-compatible API; needs GROQ_API_KEY. GROQ_BASE_URL is
# overridable only for testing against a fake server.
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_GPT_OSS_MODEL = "openai/gpt-oss-120b"


def create_model(model_name, model_dir, use_open_model_api=True, **kwargs):
    if not use_open_model_api:
        if model_name == 'mistral7b':
            return VLLMModel('Mistral-7B-Instruct-v0.2',model_dir,MISTRAL_TMPL,**kwargs) 
        elif model_name == 'deepseek7b':
            return VLLMModel('DeepSeek-R1-Distill-Qwen-7B',model_dir,DEEPSEEK_TMPL,**kwargs)
        elif model_name == 'llama3b':
            return VLLMModel('Llama-3.2-3B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
            # return HFModel('Llama-3.2-3B-Instruct',model_dir,LLAMA_TMPL,**kwargs)
        elif model_name == 'llama1b':
            return VLLMModel('Llama-3.2-1B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
        elif model_name == 'llama8b':
            return VLLMModel('Llama-3.1-8B-Instruct',model_dir,LLAMA_TMPL,**kwargs) 
        elif model_name == 'vicuna7b':
            return VLLMModel('vicuna-7b-v1.5',model_dir,VICUNA_TMPL,**kwargs)  
        elif model_name == 'mixtral8x7b':
            return VLLMModel('Mixtral-8x7B-Instruct-v0.1',model_dir,MISTRAL_TMPL,**kwargs)
        elif model_name == 'o1-mini':
            return GPTModel('o1-mini', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o':
            return GPTModel('gpt-4o', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o-mini':
            return GPTModel('gpt-4o-mini', GPT_TMPL, **kwargs)
        elif model_name == 'gpt-oss-120b':
            return SharedVLLMModel('openai/gpt-oss-120b', GPT_TMPL, **kwargs)
        elif model_name == 'qwen2.5-32b':
            return _make_qwen_model(**kwargs)
        elif model_name == 'gpt-oss-120b-local':
            return SharedVLLMModel('openai/gpt-oss-120b', GPT_TMPL,
                                   server_file=LOCAL_VLLM_SERVER_FILE, **kwargs)
        elif model_name == 'gpt-oss-120b-groq':
            return GroqModel(GROQ_GPT_OSS_MODEL, GPT_TMPL, **kwargs)
        else:
            raise NotImplementedError
    else:
        if model_name == 'tai_mistral7b':
            return TogetherAIModel('mistralai/Mistral-7B-Instruct-v0.2',MISTRAL_TMPL,**kwargs)  
        elif model_name == 'tai_llama8b':
            return TogetherAIModel('meta-llama/Llama-3-8b-chat-hf',LLAMA_TMPL,**kwargs) 
        elif model_name == 'o1-mini':
            return GPTModel('o1-mini', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o':
            return GPTModel('gpt-4o', GPT_TMPL, **kwargs) 
        elif model_name == 'gpt-4o-mini':
            return GPTModel('gpt-4o-mini', GPT_TMPL, **kwargs)
        elif model_name == 'gpt-oss-120b':
            return SharedVLLMModel('openai/gpt-oss-120b', GPT_TMPL, **kwargs)
        elif model_name == 'qwen2.5-32b':
            return _make_qwen_model(**kwargs)
        elif model_name == 'gpt-oss-120b-local':
            return SharedVLLMModel('openai/gpt-oss-120b', GPT_TMPL,
                                   server_file=LOCAL_VLLM_SERVER_FILE, **kwargs)
        elif model_name == 'gpt-oss-120b-groq':
            return GroqModel(GROQ_GPT_OSS_MODEL, GPT_TMPL, **kwargs)
        else:
            raise NotImplementedError


def _make_qwen_model(**kwargs):
    # Qwen2.5-32B-Instruct is a plain (non-reasoning) chat model: greedy
    # decoding (temperature 0.0, matching the server's example client) and the
    # repo's standard short-answer budget (MAX_NEW_TOKENS) are appropriate;
    # callers such as biogen still override max_output_tokens via kwargs.
    return SharedVLLMModel(
        'Qwen/Qwen2.5-32B-Instruct', GPT_TMPL,
        server_file=QWEN_VLLM_SERVER_FILE,
        temperature=0.0,
        default_max_output_tokens=MAX_NEW_TOKENS,
        **kwargs,
    )


class BaseModel:
    def __init__(self,cache_path=None):
        # setup the LLM response cache if cache_path is not None
        self.use_cache = cache_path is not None 
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.cache = self.load_cache()
        else:
            self.cache = {}
        self.hash = lambda x: x # for now directly string as the hash key
        # cache is a dict {hash(s):LLM response for input s}
        # 
        # end of sentence str
        self.clean_str = []

        # input prompt template
        self.prompt_template = {}

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # key under which a prompt's reasoning trace is kept in self.cache (tuple
    # keys never collide with the plain-string prompt keys)
    @staticmethod
    def _reasoning_key(prompt):
        return ('__reasoning__', prompt)

    def _record_cache_hit(self, prompt, result):
        json_trace_log.record_call(prompt, result, thinking=self.cache.get(self._reasoning_key(prompt)),
                                   cached=True)

    def query(self, prompt):
        if self.use_cache: # use cache if cache hits
            result = self.query_from_cache(prompt)
            if len(result)>0:
                self._record_cache_hit(prompt, result)
                return result

        # otherwise, do normal LLM query
        result = self._query(prompt)

        # store the response to self.cache
        if self.use_cache:
            self.cache[self.hash(prompt)]=result
        return result

    def _query(self,prompt): # will be implemented in each subclass
        raise NotImplementedError

    def batch_query(self, prompt_list):
        if self.use_cache: # use cache if cache hits
            results = self.batch_query_from_cache(prompt_list)
            if len(results)>0:
                for p, r in zip(prompt_list, results):
                    self._record_cache_hit(p, r)
                return results

        # otherwise, do normal LLM batch query

        results = self._batch_query(prompt_list)

        # store the response to self.cache
        if self.use_cache:
            for p,r in zip(prompt_list,results):
                self.cache[self.hash(p)]=r
        return results

    def _batch_query(self, prompt_list):  # will be implemented in each subclass
        raise NotImplementedError

    def query_from_cache(self,prompt):
        # return cached responses
        h = self.hash(prompt)
        if h in self.cache:
            return self.cache[h]
        else:
            return ''

    def batch_query_from_cache(self,prompt_list):
        # return cached responses
        h_list = [self.hash(prompt) for prompt in prompt_list]
        if all([h in self.cache for h in h_list]):
            return [self.cache[h] for h in h_list]
        else:
            return []


    def dump_cache(self): # dump cache to disk
        # Several runs may share one cache file (a sweep, or concurrent Slurm
        # array tasks): merge with what is on disk now and replace the file
        # atomically -- under an exclusive lock where the OS has flock -- so
        # concurrent writers never corrupt it or drop each other's entries.
        lock = None
        try:
            try:
                import fcntl
                lock = open(self.cache_path + '.lock', 'w')
                fcntl.flock(lock, fcntl.LOCK_EX)
            except ImportError:  # Windows: no flock; the atomic replace still prevents corruption
                pass
            merged = {}
            if os.path.exists(self.cache_path):
                try:
                    merged = joblib.load(self.cache_path)
                except Exception as exc:
                    logger.warning(f"could not read existing cache {self.cache_path} ({exc}); overwriting it")
            merged.update(self.cache)
            tmp = f"{self.cache_path}.{os.getpid()}.tmp"
            joblib.dump(merged, tmp)
            os.replace(tmp, self.cache_path)
            self.cache = merged
        finally:
            if lock is not None:
                lock.close()
    
    def load_cache(self): # load cache from disk
        return joblib.load(self.cache_path)
        
    def _clean_response(self,response): # clean response based on self.clean_str
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx!=-1:
                response = response[:idx]
        return response.strip()

    def query_biogen(self, prompt):
        raise NotImplementedError

    def wrap_prompt(self,data_item,as_multi_choice=True,hints=None,seperate=False):  
        # use data_item and generate the input prompt to the LLM

        # data_item should be the output of DataUtils.process_data_item()
        # as_multi_choice: if use it as a multiple-choice QA
        # hints: if we are using hints in the last step of the keyword aggregation
        # seperate: if True, return a list of prompts for differnet passages; otherwise, concatenate all passages and return a single prompt

        # get info
        question = data_item['question']
        topk_content = data_item['topk_content']
        choices = data_item.get('choices',[])
        use_retrieval = len(topk_content) > 0 # if we use retrieved passage

        def fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints):
            filling = {'query_str': question}
            if use_retrieval:
                filling.update({'context_str': context_str})
            if as_multi_choice:
                filling.update({
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3]                
                    })
            if hints is not None:
                filling.update({'hints':hints})
            return template.format(**filling)

        # get corresponding template
        mode = 'qa'
        if as_multi_choice: mode += '-mc'
        if "long_gen" in data_item: mode += '-long'
        if not use_retrieval: mode += '-zero'
        if "decode" in data_item: mode += '-decode' 
        if "genhint" in data_item: mode += '-genhint'
        if hints is not None: mode += '-hint' 
        template = self.prompt_template[mode]

        if seperate:
            return [fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints) for context_str in topk_content]
        else:
            context_str = '\n\n'.join(topk_content)
            return fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints)

    def get_token_count(self):
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
        }

    def reset_token_count(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

class VLLMModel(BaseModel):
    def __init__(self, model_name, model_dir, prompt_template, cache_path=None, max_output_tokens=None, seed=42, **kwargs):
        super().__init__(cache_path)

        from vllm import LLM, SamplingParams  # lazy import: vllm is GPU-only and heavy

        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.model_name = model_name
        self.prompt_template = prompt_template

        self.llm = LLM(
            model=model_dir + model_name,
            gpu_memory_utilization=0.5
        )
        self.sampling_params = SamplingParams(
            # max_tokens=2048,
            temperature=0
        )
        
        self.clean_str = ['\n\n']

    def _query(self, prompt):
        outputs = self.llm.generate(prompt, self.sampling_params)
        result = outputs[0].outputs[0].text
        self.total_input_tokens += outputs[0].prompt_token_ids.__len__()
        self.total_output_tokens += outputs[0].outputs[0].token_ids.__len__()
        result = self._clean_response(result)
        return result

    def _batch_query(self, prompt_list):
        results = []
        outputs = self.llm.generate(prompt_list, self.sampling_params)
        for output in outputs:
            self.total_input_tokens += len(output.prompt_token_ids)
            self.total_output_tokens += len(output.outputs[0].token_ids)
            result = self._clean_response(output.outputs[0].text)
            results.append(result)
        return results

class HFModel(BaseModel):
    def __init__(self, model_name, model_dir, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        # set max number of output tokens
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 

        # set up tokenizer and model
        model_path = model_dir + model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) 
        if "CohereForAI" in model_name:
            #'CohereForCausalLM' object has no attribute 'torch_dtype'
            self.model = AutoModelForCausalLM.from_pretrained(model_path,**kwargs) 
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda',**kwargs)
        
        self.prompt_template = prompt_template
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        with open("configs/ds_config.json", 'r') as f:
            ds_config = json.load(f)

        self.generation_kwargs = {
            'max_new_tokens':self.max_output_tokens,
            'pad_token_id':self.tokenizer.eos_token_id,
            'do_sample':True
        }
        if 'Llama-3' in model_name:
            self.generation_kwargs['eos_token_id']=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.clean_str = ['\n\n'] 


    def _query(self, prompt):
        # get text prompt as input and return text responses
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        token_length = inputs.input_ids.size(1)
        self.total_input_tokens += token_length
        # check if the num of token exceeds the max context window ..
        # this only happens when we use the bio generation data without any defense (i.e., we concatenate all long passages together)
        # we did not implement this cut off for _batch_query
        if token_length >= (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens):
            prompt_length = len(prompt)
            ratio = (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens-100)/token_length # this is a rough cut off, 100 is a buffer
            ## this is left cut 
            # prompt_cut = prompt[:int(prompt_length*ratio)] + "..." + "\n\n" + prompt[prompt.rfind("Query: Tell me a bio of"):] 
            ## this is right cut 
            prompt_cut = "Context information is below.\n" + "---------------------\n ..."+ prompt[-int(prompt_length*ratio):]
            inputs = self.tokenizer(prompt_cut, return_tensors="pt").to("cuda")
            logger.warning(f"Prompt length exceeds the limit, cut the prompt to {inputs.input_ids.size(1)} tokens")
        with torch.no_grad():
            outputs = self.model.generate(**inputs,**self.generation_kwargs)
        output_token_count = outputs[0].size(0) - inputs.input_ids.size(1)
        self.total_output_tokens += output_token_count
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
        

    def _batch_query(self, prompt_list):
        try:
            inputs = self.tokenizer(
                prompt_list, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_kwargs)
            self.total_output_tokens += outputs.numel() - inputs['input_ids'].numel()
            results = self.tokenizer.batch_decode(
                outputs[:, len(inputs[0]):], skip_special_tokens=True
            )
            results = [self._clean_response(x) for x in results]
        except Exception as e:
            print(e)
            results = []
            for prompt in prompt_list:
                results.append(self._query(prompt))
        return results


class GPTModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None,max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def _query(self, prompt: str) -> str:
        fallback = "I don't know"

        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ]

            # o1/o3-mini style reasoning models don't accept temperature and use
            # max_completion_tokens instead of max_tokens.
            if self.model_name in ("o1-mini", "o1", "o3-mini"):
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_output_tokens,
                )
            else:
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )

            # ---- defensive checks -------------------------------------------
            if chat is None or not chat.choices:
                return fallback

            content = chat.choices[0].message.content
            if not content:                       # covers None or empty string
                return fallback

            # ---- token bookkeeping (only if usage is available) ------------
            if getattr(chat, "usage", None):
                self.total_input_tokens  += getattr(chat.usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(chat.usage, "completion_tokens", 0)

            return content

        except Exception as exc:
            print("OpenAI query failed:", exc)
            return fallback


    # def _batch_query(self, prompt_list):
    #     prompt_list_with_template = [
    #         [
    #             {"role": "system", "content": "You are a helpful assistant."},
    #             {"role": "user", "content": prompt}
    #         ]
    #         for prompt in prompt_list
    #     ]
    #     try:
    #         result = batch_completion(
    #             model=self.model_name,
    #             messages=prompt_list_with_template,
    #             max_tokens=self.max_output_tokens
    #         )
    #         self.total_input_tokens += sum(x.usage.prompt_tokens for x in result)
    #         self.total_output_tokens += sum(x.usage.completion_tokens for x in result)
    #         response_list = [x.choices[0].message.content for x in result]
    #     except Exception as e:
    #         print("Error during batch_completion:", e)
    #         response_list = []
    #         for prompt in prompt_list:
    #             response_list.append(self._query(prompt))

    #     return response_list

    def _batch_query(self, prompt_list):
        return [self._query(p) for p in prompt_list]

    def query_biogen(self, prompt):
        return self.query(prompt)


class SharedVLLMModel(BaseModel):
    """Talks to a shared, already-running vLLM server over its OpenAI-compatible
    API (see CONNECT_TO_SHARED_VLLM.md). We do not start or manage the server;
    we just look up where it currently is and send requests to it.

    The server's node:port is read fresh from SHARED_VLLM_SERVER_FILE on every
    query (not cached), since the owner may restart it on a different node at
    any time.
    """

    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None,
                 server_file=None, temperature=0.7, default_max_output_tokens=2048, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        # Defaults suit gpt-oss, a reasoning model whose hidden reasoning tokens
        # share the max_tokens budget (a small value returns empty/truncated
        # answers). Non-reasoning models pass a smaller default_max_output_tokens.
        self.max_output_tokens = default_max_output_tokens if max_output_tokens is None else max_output_tokens
        self.server_file = server_file or SHARED_VLLM_SERVER_FILE
        # gpt-oss returns a reasoning trace with each answer; log it (see
        # src/reasoning_log.py). Non-reasoning models (Qwen) have none.
        self.log_reasoning = 'gpt-oss' in model_name

    def _base_url(self):
        with open(self.server_file, 'r') as f:
            node_port = f.read().strip()
        return f"http://{node_port}/v1"

    def _client(self):
        # Rebuilt on every call since the node:port can change if the server
        # was restarted (see SHARED_VLLM_SERVER_FILE handling above).
        return OpenAI(base_url=self._base_url(), api_key="dummy")

    def _query(self, prompt):
        fallback = "I don't know"
        try:
            client = self._client()
            chat = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            # extracted before the empty-answer fallback, so traces that used up
            # the whole token budget without answering are still logged
            thinking = log_reasoning("answer", self.model_name, prompt, chat) if self.log_reasoning else None
            if thinking and self.use_cache:
                self.cache[self._reasoning_key(prompt)] = thinking
            content = chat.choices[0].message.content if chat is not None and chat.choices else None
            if content and getattr(chat, "usage", None):
                self.total_input_tokens += getattr(chat.usage, "prompt_tokens", 0)
                self.total_output_tokens += getattr(chat.usage, "completion_tokens", 0)
            answer = content if content else fallback
            json_trace_log.record_call(prompt, answer, thinking=thinking, usage=usage_dict(chat),
                                       error=None if content else "empty answer (fallback used)")
            return answer
        except FileNotFoundError:
            logger.warning(f"Shared vLLM server file not found at {self.server_file}; is the server running?")
            json_trace_log.record_call(prompt, fallback, error=f"server file not found: {self.server_file}")
            return fallback
        except Exception as exc:
            print("Shared vLLM query failed:", exc)
            json_trace_log.record_call(prompt, fallback, error=f"query failed: {exc}")
            return fallback

    def _batch_query(self, prompt_list):
        # Keep concurrency modest (usage rule: ~8 parallel requests or fewer,
        # GPUs are shared). Sequential calls stay well within that.
        return [self._query(p) for p in prompt_list]


class GroqModel(SharedVLLMModel):
    """gpt-oss-120b via Groq's OpenAI-compatible API, for local testing.

    Same request shape and defaults as the shared-server gpt-oss config
    (temperature 0.7, 2048 max tokens), so local results are comparable; only
    the endpoint and auth differ. Rate-limit (429) and transient errors are
    retried with backoff by the OpenAI client itself (honours Retry-After).
    """

    def __init__(self, model_name, prompt_template, **kwargs):
        super().__init__(model_name, prompt_template, **kwargs)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set (export GROQ_API_KEY=... before running)")
        self.server_file = None  # not used: fixed endpoint
        self._groq = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key,
                            max_retries=int(os.environ.get("GROQ_MAX_RETRIES", "8")),
                            timeout=120)

    def _client(self):
        return self._groq


class TogetherAIModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        from together import Together  # lazy import: only needed for TogetherAI-backed models

        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 0.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens

        self.client = Together(api_key=os.getenv("TOGETHER_API_KEY", ""))

    def _query(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in Together _query: {e}")
            return ""

    def _batch_query(self, prompt_list):
        # TODO: implement batch query if supported
        return [self._query(prompt) for prompt in prompt_list]