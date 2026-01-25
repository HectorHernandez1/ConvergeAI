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
from utils.pdf_reader import extract_text
from utils.comparator import compare_answers, get_disagreement_summary
from pathlib import Path
from solvers.openai_solver import OpenAISolver
from solvers.anthropic_solver import AnthropicSolver

console = Console()

def sanitize_text(text: str) -> str:
    """Remove or replace problematic Unicode characters that cause encoding errors."""
    if not text:
        return text
    
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text

def extract_references() -> Optional[str]:
    """Automatically detect and extract text from references/ folder."""
    reference_dir = Path("references")
    
    if not reference_dir.exists():
        return None
    
    all_files = list(reference_dir.glob("*"))
    doc_files = [f for f in all_files if f.suffix.lower() in ['.pdf', '.ppt', '.pptx']]
    
    if not doc_files:
        return None
    
    console.print(f"[dim]Found {len(doc_files)} reference document(s) in references/[/dim]")
    
    combined_texts = []
    for doc_file in sorted(doc_files):
        try:
            text = extract_text(str(doc_file))
            text = sanitize_text(text)
            combined_texts.append(f"# Reference: {doc_file.name}\n\n{text}")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract {doc_file.name}: {e}[/yellow]")
    
    if combined_texts:
        return "\n\n".join(combined_texts)
    
    return None

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
                       max_cost: float = None) -> FinalOutput:
    max_iterations = max_iterations or settings.max_iterations
    early_stop = early_stop_threshold if early_stop_threshold is not None else settings.early_stop_threshold
    cost_limit = max_cost if max_cost is not None else settings.max_cost_usd
    
    console.print(f"[bold blue]Processing problem:[/bold blue] {problem_path}")
    problem_text = sanitize_text(extract_text(problem_path))
    references_text = extract_references()  # Extracts from references/ folder (PDFs and PPTs)
    
    openai_solver = OpenAISolver()
    anthropic_solver = AnthropicSolver()
    
    iteration_logs = []
    total_cost = 0.0
    model_responses = {"OpenAI": [], "Anthropic": []}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for iteration in range(1, max_iterations + 1):
            task = progress.add_task(f"[cyan]Iteration {iteration}/{max_iterations}...", total=None)
            
            _check_cost_warning(total_cost, cost_limit, iteration)
            
            if iteration == 1:
                response_a, response_b = await asyncio.gather(
                    openai_solver.solve(problem_text, references_text, iteration=iteration),
                    anthropic_solver.solve(problem_text, references_text, iteration=iteration)
                )
            else:
                previous_comparison = iteration_logs[-1].comparison
                disagreement_summary = get_disagreement_summary(previous_comparison.differing_questions)
                
                response_a, response_b = await asyncio.gather(
                    openai_solver.solve(problem_text, references_text, 
                                      previous_answers={
                                          "your_answers": [a.model_dump() for a in model_responses["OpenAI"][-1].answers],
                                          "other_answers": [a.model_dump() for a in model_responses["Anthropic"][-1].answers],
                                          "disagreement_summary": disagreement_summary
                                      }, 
                                      iteration=iteration),
                    anthropic_solver.solve(problem_text, references_text,
                                         previous_answers={
                                             "your_answers": [a.model_dump() for a in model_responses["Anthropic"][-1].answers],
                                             "other_answers": [a.model_dump() for a in model_responses["OpenAI"][-1].answers],
                                             "disagreement_summary": disagreement_summary
                                         },
                                         iteration=iteration)
                )
            
            model_responses["OpenAI"].append(response_a)
            model_responses["Anthropic"].append(response_b)
            
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
    consensus_answers = _determine_consensus(model_responses, final_comparison)
    
    return FinalOutput(
        timestamp=datetime.now(),
        iterations_needed=len(iteration_logs),
        final_agreement=final_comparison.agreement_percentage,
        consensus_answers=consensus_answers,
        model_responses=model_responses,
        total_cost_usd=total_cost
    )

def _determine_consensus(model_responses: dict, comparison) -> list:
    from models import Answer
    
    consensus = []
    latest_openai = model_responses["OpenAI"][-1]
    latest_anthropic = model_responses["Anthropic"][-1]
    
    openai_answers = {a.question_number: a for a in latest_openai.answers}
    anthropic_answers = {a.question_number: a for a in latest_anthropic.answers}
    
    for q_num in comparison.matching_questions:
        consensus.append(openai_answers[q_num])
    
    for diff in comparison.differing_questions:
        q_num = diff["question_number"]
        answer_a = openai_answers.get(q_num)
        answer_b = anthropic_answers.get(q_num)
        
        if answer_a and answer_b:
            if answer_a.confidence == "high" and answer_b.confidence != "high":
                consensus.append(answer_a)
            elif answer_b.confidence == "high" and answer_a.confidence != "high":
                consensus.append(answer_b)
            else:
                consensus.append(answer_a)
        elif answer_a:
            consensus.append(answer_a)
        elif answer_b:
            consensus.append(answer_b)
    
    return sorted(consensus, key=lambda x: x.question_number)

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
    parser.add_argument("--problem", help="Path to problem PDF (optional - will use first PDF in input/ directory)")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations")
    parser.add_argument("--early-stop-threshold", type=float, help="Early stop threshold (0.0-1.0, default: 0.90)")
    parser.add_argument("--max-cost", type=float, help="Maximum cost in USD (default: 5.0)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed iteration output")
    
    args = parser.parse_args()
    
    problem_path = args.problem
    
    if not problem_path:
        input_dir = Path(settings.input_dir)
        pdf_files = list(input_dir.glob("*.pdf"))
        
        if not pdf_files:
            console.print(f"[bold red]Error:[/bold red] No PDF files found in {input_dir}")
            console.print("Please add a PDF to input/ directory or specify --problem")
            return
        
        problem_path = str(pdf_files[0])
        if len(pdf_files) > 1:
            console.print(f"[yellow]Found {len(pdf_files)} PDF files. Using: {Path(problem_path).name}[/yellow]")
    
    if not os.path.exists(problem_path):
        console.print(f"[bold red]Error:[/bold red] Problem file not found: {problem_path}")
        return
    
    console.print("[bold blue]ConvergeAI[/bold blue] - AI Consensus Problem Solver")
    
    output = asyncio.run(run_consensus(problem_path, args.max_iterations, 
                                          args.early_stop_threshold, args.max_cost))
    
    save_output(output, problem_path)
    print_summary(output)

if __name__ == "__main__":
    main()
