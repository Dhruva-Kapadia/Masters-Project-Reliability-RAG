### ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search — see the paper: [arXiv PDF](https://www.arxiv.org/pdf/2509.23519, NeurIPS 2025)

## Repository structure

```shell
├── README.md                        # this file 
| 
├── main.py                          # entry point  
├── llm_eval.py                      # LLM-as-a-judge for long-form evaluation

| 
├── src
|   ├── dataset_utils.py              # tool for dataset -- load data; clean data; eval response
|   ├── model.py                      # LLM wrapper -- query; batched query; wrap_prompt 
|   ├── prompt_template.py            # prompt template
|   ├── defense.py                    # defense class
|   ├── attack.py                     # attack algorithm 
|   ├── helper.py                     # misc utils 
|
| 
├── data   
|   ├── realtimeqa.json               # a subset of realtimeqa
|   ├── open_nq.json                  # a (random) subset of the open nq dataset
|   ├── biogen.json                   # a subset of the biogen dataset
|   └── ...                 
├── plots/                          # plotting scripts (moved from repo root)
└── scripts/                        # slurm and helper scripts

```
## Dependencies

Tested with `torch==2.2.1` and `transformers==4.40.1`. This repository should be compatible with newer version of packages. `requirements.txt` lists other required packages (with version numbers commented out).

## Usage
```
python main.py 
--model_name: mistral7b,llama3b,gpt-4o-mini
--dataset_name: realtimeqa, realtimeqa-mc, open_nq, biogen
--top_k: 0, 5, 10, 20, etc.
--attack_method: none, Poison, PIA
--defense_method: none, voting, keyword, decoding
--alpha
--beta
--eta # NOTE!! the eta in this code is actually k\cdot\eta in the paper
--corruption_size
--subsample_iter: only used for some settings in biogen certification

--debug: add this flag to print some extra info for debugging
--save_response: add this flag to save the results(responses) for later analysis (currently more useful to bio_gen task)
--use_cache: add this flag to cache the results(responses) to avoid duplicate running 
```

### Viewing contradiction graphs (graph / MIS / sampleMIS)
Add `--log_graphs` to `main.py` (on by default in `scripts/wulver_run_one.slurm`). Each question's graph is appended to
`<run_dir>/graphs/<LOG_NAME>.jsonl`: every node's answer, every NLI-scored pair (with `p_contra`, edge or not,
and whether `--err` flipped it), the set kept by the defense, the documents in the final prompt, the poisoned ranks,
and whether the final answer was correct / the attack succeeded. Render with
```
python scripts/render_graphs.py runs3/graphs/*.jsonl                 # self-contained HTML viewer
python scripts/render_graphs.py runs3/graphs/X.jsonl --csv X.csv     # one summary row per question
python scripts/render_graphs.py runs3/graphs/X.jsonl --dot dot/ --render svg   # Graphviz files
```

### Saving the model's thinking traces (Query -> Thought -> Output)
Add `--save_traces` to `main.py`. Every LLM call made while answering a question (the per-document
answers, sampled-subset answers, keyword-hint / final aggregation query) is appended to
`<run_dir>/traces/<LOG_NAME>.jsonl` with the exact prompt, the model's reasoning (gpt-oss analysis channel,
read from vLLM's `reasoning_content` / `reasoning` field) and the visible output, plus the retrieved
contexts, poisoned ranks, final answer and grade. Run **without** `--use_cache` (cached answers have no
reasoning). `GPTOSS_REASONING_EFFORT=low|medium|high` optionally sets gpt-oss reasoning effort.
```
python main.py --model_name gpt-oss-120b --dataset_name open_nq --defense_method keyword --gamma 0.9 --max_samples 20 --save_traces
python scripts/render_traces.py traces/<LOG_NAME>.jsonl                  # readable Markdown, one section per question
python scripts/render_traces.py traces/<LOG_NAME>.jsonl --only attacked  # just the questions the attack won
python scripts/render_traces.py traces/<LOG_NAME>.jsonl --csv calls.csv  # one row per LLM call
```
On Wulver, `sbatch scripts/wulver_trace_demo.slurm` runs a 3-question demo (open_nq, keyword defense, PIA attack, top-5) and prints the rendered traces into `log/slurm-trace-<jobid>.out`; override with e.g. `sbatch --export=ALL,MAX_SAMPLES=5,DEFENSE=sampleMIS,ATTACK=none scripts/wulver_trace_demo.slurm`.

### Acknowledgements
This repository builds upon and was adapted from the upstream RobustRAG codebase maintained by inspire-group. See the original repository: [inspire-group/RobustRAG](https://github.com/inspire-group/RobustRAG).
