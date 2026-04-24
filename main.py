#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from models import FinalOutput, IterationLog
from config import settings
from typing import Tuple
from utils.file_reader import extract_text, extract_content
from utils.comparator import compare_answers, get_disagreement_summary
from pathlib import Path
from solvers import build_solver

console = Console()

def sanitize_text(text: str) -> str:
    """Remove or replace problematic Unicode characters that cause encoding errors."""
    if not text:
        return text
    
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text

def extract_references() -> Tuple[Optional[str], list]:
    """Automatically detect and extract text and images from references/ folder."""
    reference_dir = Path("references")

    if not reference_dir.exists():
        return None, []

    all_files = list(reference_dir.glob("*"))
    doc_files = [f for f in all_files if f.suffix.lower() in ['.pdf', '.ppt', '.pptx', '.html', '.htm']]

    if not doc_files:
        return None, []

    console.print(f"[dim]Found {len(doc_files)} reference document(s) in references/[/dim]")

    combined_texts = []
    combined_images = []
    for doc_file in sorted(doc_files):
        try:
            content = extract_content(str(doc_file))
            text = sanitize_text(content.text) if content.text else ""
            if text:
                combined_texts.append(f"# Reference: {doc_file.name}\n\n{text}")
            combined_images.extend(content.images)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract {doc_file.name}: {e}[/yellow]")

    text_result = "\n\n".join(combined_texts) if combined_texts else None
    return text_result, combined_images

def _check_cost_warning(current_cost: float, max_cost: float, iteration: int, is_final: bool = False):
    """Display cost warnings before running an iteration."""
    percentage = (current_cost / max_cost) * 100
    
    if percentage >= 90:
        console.print(f"[bold red]⚠️  WARNING: Spent ${current_cost:.2f} ({percentage:.0f}%) of ${max_cost:.2f} budget![/bold red]")
    elif percentage >= 75:
        console.print(f"[bold yellow]⚠️  WARNING: Spent ${current_cost:.2f} ({percentage:.0f}%) of ${max_cost:.2f} budget[/bold yellow]")
    elif percentage >= 50:
        console.print(f"[bold yellow]ℹ️  INFO: Spent ${current_cost:.2f} ({percentage:.0f}%) of ${max_cost:.2f} budget[/bold yellow]")
    elif not is_final and iteration > 1:
        console.print(f"[dim]💰 Running iteration {iteration} | Total cost so far: ${current_cost:.4f}[/dim]")

async def run_consensus(problem_path: str,
                       max_iterations: int = None,
                       early_stop_threshold: float = None,
                       max_cost: float = None,
                       solver_specs: Optional[list] = None) -> FinalOutput:
    max_iterations = max_iterations or settings.max_iterations
    early_stop = early_stop_threshold if early_stop_threshold is not None else settings.early_stop_threshold
    cost_limit = max_cost if max_cost is not None else settings.max_cost_usd
    specs = solver_specs or settings.solvers
    if len(specs) != 2:
        raise ValueError(f"Exactly 2 solvers required, got {len(specs)}: {specs}")

    console.print(f"[bold blue]Processing problem:[/bold blue] {problem_path}")
    content = extract_content(problem_path)
    problem_text = sanitize_text(content.text) if content.text else ""
    problem_images = content.images

    if not problem_text and problem_images:
        problem_text = "Please analyze the attached image(s) and solve any problems or answer any questions shown."

    references_text, reference_images = extract_references()
    all_images = (problem_images + reference_images) or None
    if all_images:
        console.print(f"[dim]Sending {len(all_images)} image(s) to vision APIs[/dim]")

    solver_a = build_solver(specs[0])
    solver_b = build_solver(specs[1])
    name_a, name_b = solver_a.short_name, solver_b.short_name
    console.print(f"[dim]Solvers: {solver_a.model_name} ↔ {solver_b.model_name}[/dim]")

    iteration_logs = []
    total_cost = 0.0
    model_responses = {name_a: [], name_b: []}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for iteration in range(1, max_iterations + 1):
            task = progress.add_task(f"[cyan]Iteration {iteration}/{max_iterations}...", total=None)

            _check_cost_warning(total_cost, cost_limit, iteration)

            if iteration == 1:
                response_a, response_b = await asyncio.gather(
                    solver_a.solve(problem_text, references_text, iteration=iteration, images=all_images),
                    solver_b.solve(problem_text, references_text, iteration=iteration, images=all_images)
                )
            else:
                previous_comparison = iteration_logs[-1].comparison
                disagreement_summary = get_disagreement_summary(
                    previous_comparison.differing_questions, name_a, name_b
                )

                response_a, response_b = await asyncio.gather(
                    solver_a.solve(problem_text, references_text,
                                   previous_answers={
                                       "your_answers": [a.model_dump() for a in model_responses[name_a][-1].answers],
                                       "other_answers": [a.model_dump() for a in model_responses[name_b][-1].answers],
                                       "disagreement_summary": disagreement_summary
                                   },
                                   iteration=iteration,
                                   images=all_images),
                    solver_b.solve(problem_text, references_text,
                                   previous_answers={
                                       "your_answers": [a.model_dump() for a in model_responses[name_b][-1].answers],
                                       "other_answers": [a.model_dump() for a in model_responses[name_a][-1].answers],
                                       "disagreement_summary": disagreement_summary
                                   },
                                   iteration=iteration,
                                   images=all_images)
                )

            model_responses[name_a].append(response_a)
            model_responses[name_b].append(response_b)

            total_cost += response_a.cost_usd + response_b.cost_usd

            if total_cost > cost_limit:
                console.print(f"[bold red]💰 Cost limit (${cost_limit:.2f}) exceeded! Stopping early.[/bold red]")
                break

            comparison = compare_answers(response_a, response_b)
            iteration_log = IterationLog(
                iteration=iteration,
                comparison=comparison,
                responses=[response_a, response_b],
                timestamp=datetime.now()
            )
            iteration_logs.append(iteration_log)

            total_questions = len(comparison.matching_questions) + len(comparison.differing_questions)
            progress.update(task, description=f"[cyan]Iteration {iteration}/{max_iterations}: {comparison.agreement_percentage:.1f}% agreement | {len(comparison.matching_questions)}/{total_questions} questions in consensus[/cyan]")

            if comparison.agreement_percentage >= (settings.agreement_threshold * 100):
                console.print(f"[bold green]✓ 100% agreement reached at iteration {iteration}![/bold green]")
                break

            if comparison.agreement_percentage >= (early_stop * 100):
                console.print(f"[bold green]✓ {early_stop*100:.0f}% agreement threshold reached at iteration {iteration}![/bold green]")
                break

            if iteration >= max_iterations:
                console.print(f"[bold yellow]⚠️  Reached max iterations ({max_iterations}) with {comparison.agreement_percentage:.1f}% agreement[/bold yellow]")

    final_comparison = iteration_logs[-1].comparison
    consensus_answers = _determine_consensus(model_responses, final_comparison, name_a, name_b)

    return FinalOutput(
        timestamp=datetime.now(),
        iterations_needed=len(iteration_logs),
        final_agreement=final_comparison.agreement_percentage,
        consensus_answers=consensus_answers,
        model_responses=model_responses,
        iteration_comparisons=[log.comparison for log in iteration_logs],
        total_cost_usd=total_cost
    )

_CONFIDENCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def _determine_consensus(model_responses: dict, comparison, name_a: str, name_b: str) -> list:
    """Pick a final answer for each question.

    For agreed questions: take the matching answer.

    For disagreed questions we used to just take the latest answer from
    whichever model had higher confidence. That pathology shows up hard on
    local models that flip-flop between iterations — the "latest" answer is
    often a fresh reversal, not a settled position.

    The sticky rule: look across ALL iterations of BOTH models, group
    semantically-equivalent answers together, and pick the one that was held
    the most iteration-slots. Ties broken by confidence (weighted across
    occurrences), then by preferring the latest iteration of solver A.
    """
    from utils.comparator import _check_match

    consensus = []
    histories = {
        name_a: model_responses[name_a],
        name_b: model_responses[name_b],
    }
    latest_a = {a.question_number: a for a in model_responses[name_a][-1].answers}
    latest_b = {a.question_number: a for a in model_responses[name_b][-1].answers}

    for q_num in comparison.matching_questions:
        consensus.append(latest_a.get(q_num) or latest_b.get(q_num))

    for diff in comparison.differing_questions:
        q_num = diff["question_number"]
        pick = _pick_sticky_answer(q_num, histories, name_a, _check_match)
        if pick is not None:
            consensus.append(pick)

    return sorted([c for c in consensus if c is not None], key=lambda x: x.question_number)


def _pick_sticky_answer(q_num: int, histories: dict, name_a: str, match_fn):
    """For a disagreed question, find the answer held most consistently across
    all iterations of both models. Group answers by semantic equivalence using
    match_fn (exact/numerical/semantic)."""
    # Collect every (model, iteration_index, Answer) tuple that addresses q_num
    occurrences = []
    for model_name, responses in histories.items():
        for iter_idx, resp in enumerate(responses):
            for ans in resp.answers:
                if ans.question_number == q_num:
                    occurrences.append((model_name, iter_idx, ans))
                    break

    if not occurrences:
        return None

    # Group semantically-equivalent answers into buckets
    buckets = []  # list of dicts: {"rep": Answer, "members": [(model, iter, ans)]}
    for occ in occurrences:
        _, _, ans = occ
        placed = False
        for bucket in buckets:
            if match_fn(ans.answer, bucket["rep"].answer):
                bucket["members"].append(occ)
                placed = True
                break
        if not placed:
            buckets.append({"rep": ans, "members": [occ]})

    # Score each bucket: persistence (count) + confidence (sum of weights)
    def score(bucket):
        count = len(bucket["members"])
        conf_sum = sum(_CONFIDENCE_WEIGHT.get(m[2].confidence, 0) for m in bucket["members"])
        latest_iter = max(m[1] for m in bucket["members"])
        has_a_latest = any(m[0] == name_a and m[1] == latest_iter for m in bucket["members"])
        return (count, conf_sum, latest_iter, 1 if has_a_latest else 0)

    buckets.sort(key=score, reverse=True)
    winner = buckets[0]

    # Use the representative from the most recent occurrence of the winning bucket
    latest_member = max(winner["members"], key=lambda m: m[1])
    return latest_member[2]

def save_output(output: FinalOutput, problem_path: str):
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem_name = Path(problem_path).stem
    
    json_path = output_dir / f"{problem_name}_solutions_{timestamp}.json"
    md_path = output_dir / f"{problem_name}_solutions_{timestamp}.md"
    
    with open(json_path, 'w') as f:
        f.write(output.model_dump_json(indent=2))
    
    md_content = _generate_markdown(output, problem_name)
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    console.print(f"\n[bold green]Results saved:[/bold green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  Markdown: {md_path}")

def _generate_markdown(output: FinalOutput, problem_name: str) -> str:
    lines = [
        f"# ConvergeAI Solution Report",
        f"**Problem:** {problem_name}",
        f"**Generated:** {output.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Consensus Reached:** Iteration {output.iterations_needed} of {settings.max_iterations}",
        f"**Agreement:** {output.final_agreement:.1f}%",
        f"**Total Cost:** ${output.total_cost_usd:.2f}",
        ""
    ]
    
    for answer in output.consensus_answers:
        lines.extend([
            "---",
            f"## Question {answer.question_number}",
            f"**Question:** {answer.question_text}",
            f"**Consensus Answer:** {answer.answer}",
            f"**Confidence:** {answer.confidence.capitalize()}",
            f"**Reasoning:** {answer.reasoning}",
        ])
        if answer.references_cited:
            lines.append(f"**References Cited:** {', '.join(answer.references_cited)}")
        lines.append("")
    
    # Disagreement Tracker section
    comparisons = output.iteration_comparisons
    if not comparisons and len(output.model_responses) == 2:
        # Backward compatibility: recompute from model_responses
        from utils.comparator import compare_answers
        model_names = list(output.model_responses.keys())
        for i in range(len(output.model_responses[model_names[0]])):
            comp = compare_answers(
                output.model_responses[model_names[0]][i],
                output.model_responses[model_names[1]][i]
            )
            comparisons.append(comp)

    if comparisons:
        model_names = list(output.model_responses.keys())
        num_questions = len(comparisons[0].matching_questions) + len(comparisons[0].differing_questions)

        lines.extend(["---", "## Disagreement Tracker", ""])

        # Agreement progression table
        lines.append("### Agreement Progression")
        lines.append("")
        header = "| Iteration | Agreement |"
        for q in range(1, num_questions + 1):
            header += f" Q{q} |"
        lines.append(header)
        sep = "|-----------|-----------|"
        for _ in range(num_questions):
            sep += "----|"
        lines.append(sep)

        for idx, comp in enumerate(comparisons):
            row = f"| {idx + 1} | {comp.agreement_percentage:.1f}% |"
            all_questions = set(comp.matching_questions) | {d["question_number"] for d in comp.differing_questions}
            for q in range(1, num_questions + 1):
                if q in comp.matching_questions:
                    row += " :white_check_mark: |"
                elif q in all_questions:
                    row += " :x: |"
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

        # Detailed disagreements per iteration
        lines.append("### Iteration Deltas")
        lines.append("")

        for idx, comp in enumerate(comparisons):
            if not comp.differing_questions:
                continue
            lines.append(f"#### Iteration {idx + 1} — {comp.agreement_percentage:.1f}% agreement ({len(comp.differing_questions)} disagreement{'s' if len(comp.differing_questions) != 1 else ''})")
            lines.append("")
            for diff in comp.differing_questions:
                q_num = diff["question_number"]
                q_text = diff.get("question_text", "")
                lines.append(f"**Question {q_num}:** {q_text}")
                lines.append("")

                answer_a = diff.get("answer_a")
                answer_b = diff.get("answer_b")
                a_text = answer_a.get("answer", "N/A") if answer_a else "N/A"
                b_text = answer_b.get("answer", "N/A") if answer_b else "N/A"

                lines.append(f"| Model | Answer |")
                lines.append(f"|-------|--------|")
                lines.append(f"| {model_names[0]} | {a_text} |")
                lines.append(f"| {model_names[1]} | {b_text} |")
                lines.append("")

        # Unresolved disagreements summary
        final_comp = comparisons[-1]
        if final_comp.differing_questions:
            lines.append("### Unresolved Disagreements")
            lines.append("")
            lines.append(f"The following {len(final_comp.differing_questions)} question(s) never reached agreement after {len(comparisons)} iteration(s):")
            lines.append("")
            for diff in final_comp.differing_questions:
                q_num = diff["question_number"]
                answer_a = diff.get("answer_a")
                answer_b = diff.get("answer_b")
                a_text = answer_a.get("answer", "N/A") if answer_a else "N/A"
                b_text = answer_b.get("answer", "N/A") if answer_b else "N/A"
                lines.append(f"- **Q{q_num}**: {model_names[0]} = \"{a_text}\" vs {model_names[1]} = \"{b_text}\"")
            lines.append("")

    # Iteration History section
    lines.extend([
        "---",
        "## Iteration History",
        ""
    ])

    for model_name in output.model_responses:
        model_total_cost = sum(r.cost_usd for r in output.model_responses[model_name])
        model_total_tokens = sum(r.tokens_used for r in output.model_responses[model_name])
        lines.append(f"### {model_name} (${model_total_cost:.4f} total, {model_total_tokens} total tokens)")
        lines.append("")
        for response in output.model_responses[model_name]:
            lines.append(f"- Iteration {response.iteration}: {response.tokens_used} tokens, ${response.cost_usd:.4f}")
        lines.append("")

    return "\n".join(lines)

def print_summary(output: FinalOutput):
    console.print("\n[bold]Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Iterations", str(output.iterations_needed))
    table.add_row("Final Agreement", f"{output.final_agreement:.1f}%")
    table.add_row("Total Cost", f"${output.total_cost_usd:.2f}")
    table.add_row("Questions Solved", str(len(output.consensus_answers)))
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="ConvergeAI - AI consensus problem solver")
    parser.add_argument("--problem", help="Path to problem file (PDF/PPT/HTML) (optional - will use first file in input/ directory)")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations")
    parser.add_argument("--early-stop-threshold", type=float, help="Early stop threshold (0.0-1.0, default: 0.90)")
    parser.add_argument("--max-cost", type=float, help="Maximum cost in USD (default: 5.0)")
    parser.add_argument("--solvers", type=str,
                        help=("Comma-separated pair of solvers. Each is 'openai', 'anthropic', "
                              "or 'ollama:<model>'. Example: "
                              "--solvers ollama:gemma4:31b,ollama:qwen3.5:35b-a3b"))
    parser.add_argument("--verbose", action="store_true", help="Show detailed iteration output")

    args = parser.parse_args()

    solver_specs = None
    if args.solvers:
        solver_specs = [s.strip() for s in args.solvers.split(",") if s.strip()]
        if len(solver_specs) != 2:
            console.print(f"[bold red]Error:[/bold red] --solvers must list exactly 2 models, got {len(solver_specs)}")
            return
    
    problem_path = args.problem
    
    if not problem_path:
        input_dir = Path(settings.input_dir)
        supported_extensions = ['.pdf', '.ppt', '.pptx', '.html', '.htm', '.png', '.jpg', '.jpeg', '.gif', '.webp']
        doc_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        
        if not doc_files:
            console.print(f"[bold red]Error:[/bold red] No supported files found in {input_dir}")
            console.print("Please add a file to input/ directory or specify --problem")
            return
        
        problem_path = str(doc_files[0])
        if len(doc_files) > 1:
            console.print(f"[yellow]Found {len(doc_files)} files. Using: {Path(problem_path).name}[/yellow]")
    
    if not os.path.exists(problem_path):
        console.print(f"[bold red]Error:[/bold red] Problem file not found: {problem_path}")
        return
    
    console.print("[bold blue]ConvergeAI[/bold blue] - AI Consensus Problem Solver")
    
    output = asyncio.run(run_consensus(problem_path, args.max_iterations,
                                          args.early_stop_threshold, args.max_cost,
                                          solver_specs=solver_specs))
    
    save_output(output, problem_path)
    print_summary(output)

if __name__ == "__main__":
    main()
