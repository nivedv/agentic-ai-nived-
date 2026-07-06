import json
import os
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Side-effect-free import of the Lab 4.1 query pipeline.
from aria_rag import answer, llm, embeddings

EVAL_PATH      = os.path.join(os.path.dirname(__file__), "meridian_eval_set.json")
PARTICIPANT_ID = int(os.environ["PARTICIPANT_ID"])

CONFIG_A = "meridian-rag"      # wizard-default chunking
CONFIG_B = "meridian-rag-b"    # ~500-char chunks (skillset maximumPageLength edited)

judge_llm = LangchainLLMWrapper(llm)
judge_emb = LangchainEmbeddingsWrapper(embeddings)
METRICS   = [faithfulness, answer_relevancy, context_recall]


def my_slice() -> list[dict]:
    """Deterministic 3-4 question slice per participant. Every question is
    covered ~8-9 times across 26 participants -> class aggregate shows judge
    variance as data, not as a claim."""
    with open(EVAL_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    return questions[(PARTICIPANT_ID - 1) % 3 :: 3]
def generate_samples(index_name: str, k: int = 5) -> list[SingleTurnSample]:
    """Run each assigned question through Lab 4.1's answer(), pinned to
    index_name. Field mapping (the whole setup):
      user_input          <- the question
      response            <- what the pipeline GENERATED
      retrieved_contexts  <- chunk texts retrieval RETURNED (never full docs)
      reference           <- hand-written ground truth (context_recall only)
    """
    samples = []
    for item in my_slice():
        ans_text, chunks = answer(item["question"], k=k, mode="hybrid",
                                  index_name=index_name)
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=ans_text,
            retrieved_contexts=[c["chunk"] for c in chunks],
            reference=item["ground_truth"],
        ))
        print(f"  [{index_name}] generated: {item['id']}")
    return samples
def run_evaluation(label: str, index_name: str) -> tuple[dict, "pd.DataFrame"]:
    print(f"\n=== {label} ({index_name}) · participant {PARTICIPANT_ID} · "
          f"questions {[q['id'] for q in my_slice()]} ===")
    samples = generate_samples(index_name)
    dataset = EvaluationDataset(samples=samples)
    result  = evaluate(dataset=dataset, metrics=METRICS,
                       llm=judge_llm, embeddings=judge_emb)
    df = result.to_pandas()
    means = {
        "label": label, "index": index_name,
        "faithfulness":     round(df["faithfulness"].mean(), 3),
        "answer_relevancy": round(df["answer_relevancy"].mean(), 3),
        "context_recall":   round(df["context_recall"].mean(), 3),
    }
    print(means)
    return means, df


if __name__ == "__main__":
    means_a, df_a = run_evaluation("Config A (wizard-default)", CONFIG_A)
    df_a.to_csv(f"ragas_a_p{PARTICIPANT_ID}.csv", index=False)

    means_b, df_b = run_evaluation("Config B (~500 chars)", CONFIG_B)
    df_b.to_csv(f"ragas_b_p{PARTICIPANT_ID}.csv", index=False)

    print("\n=== MY SLICE, A vs B ===")
    for m in ("faithfulness", "answer_relevancy", "context_recall"):
        delta = round(means_b[m] - means_a[m], 3)
        print(f"{m:<18} A={means_a[m]:<7} B={means_b[m]:<7} delta={delta:+}")
