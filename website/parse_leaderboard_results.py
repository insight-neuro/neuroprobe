#!/usr/bin/env python3
"""
Script to parse leaderboard results and generate HTML tables for the Neuroprobe website.
This script loads all submission results from the leaderboard folder, calculates overall AUROC
scores, ranks models, and updates the index.html file with the generated tables.
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import statistics

# Task mapping from the SUBMIT.md documentation
NEUROPROBE_TASKS_MAPPING = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'delta_volume': 'Delta Volume',
    'pitch': 'Voice Pitch',
    'word_index': 'Word Position',
    'word_gap': 'Inter-word Gap',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    'word_length': 'Word Length',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'frame_brightness': 'Frame Brightness',
    'face_num': 'Number of Faces',
}

class LeaderboardParser:
    def __init__(self, leaderboard_dir: str = "../leaderboard", website_dir: str = "."):
        self.leaderboard_dir = Path(leaderboard_dir)
        self.website_dir = Path(website_dir)
        
    def load_submission(self, submission_dir: Path) -> Dict[str, Any]:
        """Load a single submission's metadata and results."""
        
        # Load metadata
        metadata_file = submission_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Warning: No metadata.json found in {submission_dir}")
            return None
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Load results for each split type
        submission_data = {
            'metadata': metadata,
            'submission_dir': submission_dir.name,
            'results': {}
        }
        
        # Check for each split type
        for split_type in ['Cross-Session', 'Cross-Subject', 'Within-Session']:
            split_dir = submission_dir / split_type
            if split_dir.exists():
                submission_data['results'][split_type] = self.load_split_results(split_dir)
                
        return submission_data
    
    def load_split_results(self, split_dir: Path) -> Dict[str, Any]:
        """Load results for a specific split (Cross-Session, Cross-Subject, etc.)."""
        
        results = {}
        
        # Load all population_*.json files
        for task_file in split_dir.glob("population_*.json"):
            task_name = task_file.stem.replace("population_", "")
            
            with open(task_file, 'r') as f:
                task_data = json.load(f)
                
            # Calculate average AUROC across all sessions for this task
            auroc_scores = []
            
            for session_name, session_data in task_data.get('evaluation_results', {}).items():
                population_data = session_data.get('population', {})
                for time_bin_name, time_bin_data in population_data.items():
                    folds = time_bin_data.get('folds', [])
                    for fold in folds:
                        test_roc_auc = fold.get('test_roc_auc')
                        if test_roc_auc is not None:
                            auroc_scores.append(test_roc_auc)
            
            if auroc_scores:
                results[task_name] = {
                    'mean_auroc': statistics.mean(auroc_scores),
                    'all_auroc_scores': auroc_scores,
                    'num_evaluations': len(auroc_scores)
                }
                
        return results
    
    def calculate_overall_auroc(self, split_results: Dict[str, Any]) -> float:
        """Calculate overall AUROC by averaging across all tasks."""
        
        if not split_results:
            return 0.0
            
        task_aurocs = []
        for task_name, task_data in split_results.items():
            if 'mean_auroc' in task_data:
                task_aurocs.append(task_data['mean_auroc'])
                
        return statistics.mean(task_aurocs) if task_aurocs else 0.0
    
    def load_all_submissions(self) -> List[Dict[str, Any]]:
        """Load all submissions from the leaderboard directory."""
        
        submissions = []
        
        for submission_dir in self.leaderboard_dir.iterdir():
            if submission_dir.is_dir():
                submission_data = self.load_submission(submission_dir)
                if submission_data:
                    submissions.append(submission_data)
                    
        return submissions
    
    def rank_submissions(self, submissions: List[Dict[str, Any]], split_type: str) -> List[Dict[str, Any]]:
        """Rank submissions by overall AUROC for a specific split type."""
        
        # Filter submissions that have results for this split type
        valid_submissions = []
        for submission in submissions:
            if split_type in submission['results']:
                overall_auroc = self.calculate_overall_auroc(submission['results'][split_type])
                submission_copy = submission.copy()
                submission_copy['overall_auroc'] = overall_auroc
                valid_submissions.append(submission_copy)
        
        # Sort by overall AUROC (descending)
        valid_submissions.sort(key=lambda x: x['overall_auroc'], reverse=True)
        
        return valid_submissions
    
    def format_date(self, timestamp: int) -> str:
        """Format timestamp to readable date."""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        except:
            return "Unknown"
    
    def get_rank_badge_class(self, rank: int) -> str:
        """Get CSS class for rank badge."""
        if rank == 1:
            return "gold"
        elif rank == 2:
            return "silver"
        elif rank == 3:
            return "bronze"
        else:
            return ""
    
    def generate_html_table(self, ranked_submissions: List[Dict[str, Any]], split_type: str) -> str:
        """Generate HTML table for a specific split type."""
        
        if not ranked_submissions:
            return """
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Model</th>
                                    <th>Author</th>
                                    <th>Organization</th>
                                    <th>Date</th>
                                    <th><b>Overall AUROC</b></th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td colspan="6">No submissions available for this split.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
            """
        
        # Get all unique tasks across all submissions for this split
        all_tasks = set()
        for submission in ranked_submissions:
            if split_type in submission['results']:
                all_tasks.update(submission['results'][split_type].keys())
        
        # Sort tasks by the order in NEUROPROBE_TASKS_MAPPING
        task_order = list(NEUROPROBE_TASKS_MAPPING.keys())
        sorted_tasks = sorted(all_tasks, key=lambda x: task_order.index(x) if x in task_order else len(task_order))
        
        # Generate table header
        header_html = """
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>Model</th>
                                    <th>Author</th>
                                    <th>Organization</th>
                                    <th>Date</th>
                                    <th><b>Overall AUROC</b></th>"""
        
        # Add task columns
        for task in sorted_tasks:
            task_display_name = NEUROPROBE_TASKS_MAPPING.get(task, task.replace('_', ' ').title())
            header_html += f"\n                                    <th>{task_display_name}</th>"
        
        header_html += """
                                </tr>
                            </thead>
                            <tbody>"""
        
        # Generate table rows
        rows_html = ""
        for rank, submission in enumerate(ranked_submissions, 1):
            metadata = submission['metadata']
            
            # Rank badge
            rank_class = self.get_rank_badge_class(rank)
            rank_badge = f'<span class="rank-badge {rank_class}">{rank}</span>' if rank_class else str(rank)
            
            # Basic info
            model_name = metadata.get('model_name', 'Unknown')
            author = metadata.get('author', 'Unknown')
            organization = metadata.get('organization', 'Unknown')
            date = self.format_date(metadata.get('timestamp', 0))
            overall_auroc = f"{submission['overall_auroc']:.3f}"
            
            rows_html += f"""
                                <tr>
                                    <td>{rank_badge}</td>
                                    <td>{model_name}</td>
                                    <td>{author}</td>
                                    <td>{organization}</td>
                                    <td>{date}</td>
                                    <td>{overall_auroc}</td>"""
            
            # Add task-specific AUROC scores
            split_results = submission['results'].get(split_type, {})
            for task in sorted_tasks:
                if task in split_results:
                    task_auroc = f"{split_results[task]['mean_auroc']:.3f}"
                else:
                    task_auroc = "---"
                rows_html += f"\n                                    <td>{task_auroc}</td>"
            
            rows_html += "\n                                </tr>"
        
        # Close table
        footer_html = """
                            </tbody>
                        </table>
                    </div>"""
        
        return header_html + rows_html + footer_html
    
    def update_index_html(self, cross_session_table: str, cross_subject_table: str):
        """Update the index.html file with the generated tables."""
        
        index_file = self.website_dir / "index.html"
        
        with open(index_file, 'r') as f:
            html_content = f.read()
        
        # Find and replace Cross-Session table
        cross_session_start = html_content.find('<div class="tab-content active" id="population_cross_session">')
        cross_session_end = html_content.find('</div>', cross_session_start)
        cross_session_end = html_content.find('</div>', cross_session_end) # find second </div>
        
        if cross_session_start != -1 and cross_session_end != -1:
            cross_session_content = f"""<div class="tab-content active" id="population_cross_session">
{cross_session_table}
                </div>"""
            html_content = (html_content[:cross_session_start] + 
                          cross_session_content + 
                          html_content[cross_session_end + 6:])
        
        # Find and replace Cross-Subject table
        cross_subject_start = html_content.find('<div class="tab-content" id="population_cross_subject">')
        cross_subject_end = html_content.find('</div>', cross_subject_start)
        cross_subject_end = html_content.find('</div>', cross_subject_end) # find second </div>
        
        if cross_subject_start != -1 and cross_subject_end != -1:
            cross_subject_content = f"""<div class="tab-content" id="population_cross_subject">
{cross_subject_table}
                </div>"""
            html_content = (html_content[:cross_subject_start] + 
                          cross_subject_content + 
                          html_content[cross_subject_end + 6:])
        
        # Write back to file
        with open(index_file, 'w') as f:
            f.write(html_content)
    
    def run(self):
        """Main function to parse leaderboard and update website."""
        
        print("Loading all submissions...")
        submissions = self.load_all_submissions()
        print(f"Found {len(submissions)} submissions")
        
        # Rank submissions for each split type
        print("\nRanking submissions...")
        cross_session_ranked = self.rank_submissions(submissions, 'Cross-Session')
        cross_subject_ranked = self.rank_submissions(submissions, 'Cross-Subject')
        
        print(f"Cross-Session: {len(cross_session_ranked)} submissions")
        print(f"Cross-Subject: {len(cross_subject_ranked)} submissions")
        
        # Generate HTML tables
        print("\nGenerating HTML tables...")
        cross_session_table = self.generate_html_table(cross_session_ranked, 'Cross-Session')
        cross_subject_table = self.generate_html_table(cross_subject_ranked, 'Cross-Subject')
        
        # Update index.html
        print("Updating index.html...")
        self.update_index_html(cross_session_table, cross_subject_table)
        
        print("Done! Leaderboard updated successfully.")
        
        # Print summary
        print("\n=== LEADERBOARD SUMMARY ===")
        print("\nCross-Session Rankings:")
        for i, submission in enumerate(cross_session_ranked[:5], 1):
            metadata = submission['metadata']
            print(f"{i}. {metadata['model_name']} ({metadata['author']}) - {submission['overall_auroc']:.3f}")
            
        print("\nCross-Subject Rankings:")
        for i, submission in enumerate(cross_subject_ranked[:5], 1):
            metadata = submission['metadata']
            print(f"{i}. {metadata['model_name']} ({metadata['author']}) - {submission['overall_auroc']:.3f}")

if __name__ == "__main__":
    parser = LeaderboardParser()
    parser.run()
