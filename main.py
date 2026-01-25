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
from solvers.openai_solver import OpenAISolver
from solvers.anthropic_solver import AnthropicSolver

console = Console()

async def run_consensus(problem_path: str, 
                       references_path: Optional[str] = None,
                       max_iterations: int = None) -> FinalOutput:
    max_iterations = max_iterations or settings.max_iterations
    
    console.print(f"[bold blue]Processing problem:[/bold blue] {problem_path}")
    problem_text = extract_text(problem_path)
    references_text = extract_text(references_path) if references_path else None
    
    openai_solver = OpenAISolver()
    anthropic_solver = AnthropicSolver()
    
    iteration_logs = []
    total_cost = 0.0
    model_responses = {"OpenAI": [], "Anthropic": []}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for iteration in range(1, max_iterations + 1):
            task = progress.add_task(f"[cyan]Iteration {iteration}/{max_iterations}...", total=None)
            
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
            
            if total_cost > settings.max_cost_usd:
                console.print(f"[bold yellow]Cost limit (${settings.max_cost_usd}) exceeded. Stopping early.[/bold yellow]")
                break
            
            comparison = compare_answers(response_a, response_b)
            iteration_log = IterationLog(
                iteration=iteration,
                comparison=comparison,
                responses=[response_a, response_b],
                timestamp=datetime.now()
            )
            iteration_logs.append(iteration_log)
            
            progress.update(task, description=f"[cyan]Iteration {iteration}/{max_iterations} complete - {comparison.agreement_percentage:.1f}% agreement[/cyan]")
            
            if comparison.agreement_percentage >= (settings.agreement_threshold * 100):
                console.print(f"[bold green]100% agreement reached at iteration {iteration}![/bold green]")
                break
    
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
        lines.append(f"### {model_name}")
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
    parser.add_argument("--problem", required=True, help="Path to problem PDF")
    parser.add_argument("--references", help="Path to reference PDF (optional)")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations")
    parser.add_argument("--verbose", action="store_true", help="Show detailed iteration output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.problem):
        console.print(f"[bold red]Error:[/bold red] Problem file not found: {args.problem}")
        return
    
    console.print("[bold blue]ConvergeAI[/bold blue] - AI Consensus Problem Solver")
    
    output = asyncio.run(run_consensus(args.problem, args.references, args.max_iterations))
    
    save_output(output, args.problem)
    print_summary(output)

if __name__ == "__main__":
    main()
