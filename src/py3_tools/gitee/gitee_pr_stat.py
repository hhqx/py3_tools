import logging
import requests
import csv
import os
import json
import pandas as pd
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

# Configure module-level logger for debug output
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GiteePRConfig:
    """
    Configuration for querying Pull Requests from a Gitee repository.

    Attributes:
        owner (str): The user or organization owning the repository.
        repo (str): The repository name.
        author_name (str): The Gitee username of the PR author to filter by.
        state (str): PR state to include: 'open', 'closed', or 'all'. Default is 'all'.
        per_page (int): Number of PRs per API page (max 100).
        max_pages (int): Maximum number of pages to fetch.
        since (Optional[datetime]): Only include PRs created or updated after this timestamp.
        until (Optional[datetime]): Only include PRs created or updated before this timestamp.
        output_dir (str): Directory to save exported CSV and JSON files.
    """
    owner: str
    repo: str
    author_name: str
    state: str = 'all'
    per_page: int = 50
    max_pages: int = 5
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    output_dir: str = 'output'  # Add output directory configuration

class GiteePRClient:
    """
    Client to fetch PRs from Gitee based on the given configuration.
    Resolves author_name to author_id and raises an error if resolution fails.
    Provides detailed debug logging for each step.
    """

    BASE_URL = "https://gitee.com/api/v5"

    def __init__(self, config: GiteePRConfig):
        self.config = config
        self.session = requests.Session()

        # Resolve author_name to author_id
        self.author_id = self._resolve_author_id(self.config.author_name)
        if self.author_id is None:
            logger.error("Failed to resolve Gitee user ID for username '%s'", self.config.author_name)
            raise ValueError(f"Failed to resolve Gitee user ID for username '{self.config.author_name}'")
        logger.debug("Resolved author '%s' to ID %d", self.config.author_name, self.author_id)

    def _resolve_author_id(self, username: str) -> Optional[int]:
        """
        Fetch the numeric user ID for a given Gitee username.
        """
        url = f"{self.BASE_URL}/users/{username}"
        logger.debug("Requesting user info URL: %s", url)
        resp = self.session.get(url)
        if resp.status_code == 404:
            logger.debug("User '%s' not found (404)", username)
            return None
        resp.raise_for_status()
        data = resp.json()
        return data.get('id')

    def fetch_prs(self) -> List[Dict[str, Any]]:
        """
        Fetches PRs matching the configuration. Performs local filtering by author_id and time range.
        Detailed debug logs are produced for the request and filtering process.
        """
        raw_prs: List[Dict[str, Any]] = []
        for page in range(1, self.config.max_pages + 1):
            params = {
                'state': self.config.state,
                'author': self.config.author_name,
                'per_page': self.config.per_page,
                'page': page
            }
            url = f"{self.BASE_URL}/repos/{self.config.owner}/{self.config.repo}/pulls"
            logger.debug("Fetching PRs from URL: %s with params: %s", url, params)
            resp = self.session.get(url, params=params)
            logger.debug("Received status code: %d", resp.status_code)
            resp.raise_for_status()
            prs = resp.json()
            logger.debug("Page %d returned %d PRs", page, len(prs))
            if not prs:
                break
            raw_prs.extend(prs)
            if len(prs) < self.config.per_page:
                break

        logger.debug("Total PRs fetched before filtering: %d", len(raw_prs))

        # Filter by author_id and time range
        filtered: List[Dict[str, Any]] = []
        for pr in raw_prs:
            user = pr.get('user') or pr.get('author')
            pr_time_str = pr.get('updated_at') or pr.get('created_at')
            pr_time = datetime.fromisoformat(pr_time_str.rstrip('Z')) if pr_time_str else None

            if user and user.get('id') == self.author_id:
                # Convert timezone-aware datetime to naive for comparison
                if pr_time and pr_time.tzinfo:
                    pr_time_naive = pr_time.replace(tzinfo=None)
                else:
                    pr_time_naive = pr_time
                
                if self.config.since and pr_time_naive and pr_time_naive < self.config.since:
                    logger.debug("PR #%d filtered out: updated_at %s is before since %s", pr['number'], pr_time, self.config.since)
                    continue
                if self.config.until and pr_time_naive and pr_time_naive > self.config.until:
                    logger.debug("PR #%d filtered out: updated_at %s is after until %s", pr['number'], pr_time, self.config.until)
                    continue
                filtered.append(pr)

        logger.debug("Total PRs after filtering: %d", len(filtered))
        return filtered
    
    def fetch_pr_details(self, pr_number: int) -> Dict[str, Any]:
        """
        Fetch detailed information about a specific PR including file changes.
        
        Args:
            pr_number (int): The PR number to fetch details for.
            
        Returns:
            Dict[str, Any]: Detailed information about the PR including file changes.
        """
        
        url = f"{self.BASE_URL}/repos/{self.config.owner}/{self.config.repo}/pulls/{pr_number}/files"
        logger.debug(f"Fetching file changes for PR #{pr_number} from URL: {url}")
        resp = self.session.get(url)
        logger.debug(f"Received status code: {resp.status_code}")
        resp.raise_for_status()
        return resp.json()
    
    def _process_pr_files(self, files_data: Dict[str, Any], pr_number: int) -> List[Dict[str, Any]]:
        """
        Process file change data from a PR to extract relevant statistics.
        
        Args:
            files_data (Dict[str, Any]): Raw file change data from the API.
            pr_number (int): The PR number.
            
        Returns:
            List[Dict[str, Any]]: Processed file change data with statistics.
        """
        file_changes = []
        
        # import ipdb; ipdb.set_trace()  # Set a breakpoint for debugging
        if isinstance(files_data, dict) and 'diffs' in files_data:
            for diff in files_data['diffs']:
                file_name = diff.get('statistic', {}).get('path', '')
                change_type = diff.get('statistic', {}).get('type', '')
                added_lines = diff.get('added_lines', 0)
                removed_lines = diff.get('removed_lines', 0)
                
                file_changes.append({
                    'file_name': file_name,
                    'change_type': change_type,
                    'added_lines': added_lines,
                    'removed_lines': removed_lines,
                    'total_lines': added_lines + removed_lines
                })
        else:
            logger.warning(f"Unrecognized file change format for PR #{pr_number}")
        
        return file_changes
    
    def collect_pr_details(self, prs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect detailed information about PRs and their file changes.
        
        Args:
            prs (List[Dict[str, Any]]): List of PRs to process.
            
        Returns:
            List[Dict[str, Any]]: Detailed PR information with file changes.
        """
        pr_details = []
        for pr in prs:
            pr_number = pr['number']
            pr_title = pr['title']
            pr_created_at = pr.get('created_at', '')
            pr_updated_at = pr.get('updated_at', '')
            pr_merged_at = pr.get('merged_at', '')
            pr_state = pr.get('state', '')
            pr_url = pr['html_url']
            
            logger.info(f"Processing PR #{pr_number}: {pr_title}")
            
            try:
                files_data = self.fetch_pr_details(pr_number)
                
                # Process file changes based on the API response structure
                file_changes = []
                total_added = 0
                total_removed = 0
                
                if isinstance(files_data, list):
                    # Handle response format where files_data is a list of file changes
                    for file_item in files_data:
                        file_name = file_item.get('filename', '')
                        change_type = file_item.get('status', '')
                        added_lines = int(file_item.get('additions', 0))
                        removed_lines = int(file_item.get('deletions', 0))
                        total_added += added_lines
                        total_removed += removed_lines
                        
                        file_changes.append({
                            'file_name': file_name,
                            'change_type': change_type,
                            'added_lines': added_lines,
                            'removed_lines': removed_lines,
                            'total_lines': added_lines + removed_lines
                        })
                elif isinstance(files_data, dict) and 'diffs' in files_data:
                    # Handle response format with 'diffs' structure
                    total_added = files_data.get('added_lines', 0)
                    total_removed = files_data.get('removed_lines', 0)
                    
                    for diff in files_data['diffs']:
                        file_name = diff.get('statistic', {}).get('path', '')
                        change_type = diff.get('statistic', {}).get('type', '')
                        added_lines = diff.get('added_lines', 0)
                        removed_lines = diff.get('removed_lines', 0)
                        
                        file_changes.append({
                            'file_name': file_name,
                            'change_type': change_type,
                            'added_lines': added_lines,
                            'removed_lines': removed_lines,
                            'total_lines': added_lines + removed_lines
                        })
                
                total_files = len(file_changes)
                
                # Add each file change as a separate entry
                if file_changes:
                    for file_data in file_changes:
                        pr_details.append({
                            'PR Number': pr_number,
                            'PR Title': pr_title,
                            'PR State': pr_state,
                            'Created At': pr_created_at,
                            'Updated At': pr_updated_at,
                            'Merged At': pr_merged_at,
                            'PR URL': pr_url,
                            'File Name': file_data['file_name'],
                            'Change Type': file_data['change_type'],
                            'Added Lines': file_data['added_lines'],
                            'Removed Lines': file_data['removed_lines'],
                            'Total Changed Lines': file_data['total_lines'],
                            'Total Files Changed': total_files,
                            'Total PR Added Lines': total_added,
                            'Total PR Removed Lines': total_removed
                        })
                else:
                    pr_details.append({
                        'PR Number': pr_number,
                        'PR Title': pr_title,
                        'PR State': pr_state,
                        'Created At': pr_created_at,
                        'Updated At': pr_updated_at,
                        'Merged At': pr_merged_at,
                        'PR URL': pr_url,
                        'File Name': 'No files changed',
                        'Change Type': '',
                        'Added Lines': 0,
                        'Removed Lines': 0,
                        'Total Changed Lines': 0,
                        'Total Files Changed': 0,
                        'Total PR Added Lines': 0,
                        'Total PR Removed Lines': 0
                    })
                
            except Exception as e:
                logger.error(f"Failed to fetch details for PR #{pr_number}: {str(e)}")
                pr_details.append({
                    'PR Number': pr_number,
                    'PR Title': pr_title,
                    'PR State': pr_state,
                    'Created At': pr_created_at,
                    'Updated At': pr_updated_at,
                    'Merged At': pr_merged_at,
                    'PR URL': pr_url,
                    'File Name': f'ERROR: {str(e)}',
                    'Change Type': '',
                    'Added Lines': 0,
                    'Removed Lines': 0,
                    'Total Changed Lines': 0,
                    'Total Files Changed': 0,
                    'Total PR Added Lines': 0,
                    'Total PR Removed Lines': 0
                })
        
        return pr_details
    
    def export_pr_details(self, pr_details: List[Dict[str, Any]], filename: str, 
                          format: Literal['csv', 'json'] = 'csv') -> None:
        """
        Export PR details to a file in the specified format.
        
        Args:
            pr_details (List[Dict[str, Any]]): Detailed PR information with file changes.
            filename (str): The filename for output.
            format (Literal['csv', 'json']): Export format, either 'csv' or 'json'.
        """
        if not pr_details:
            logger.warning(f"No PR details to export to {filename}")
            return
            
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        if format.lower() == 'csv':
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=pr_details[0].keys())
                writer.writeheader()
                writer.writerows(pr_details)
            logger.info(f"Exported {len(pr_details)} PR file changes to CSV: {filename}")
        
        elif format.lower() == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(pr_details, f, ensure_ascii=False, indent=2)
            logger.info(f"Exported {len(pr_details)} PR file changes to JSON: {filename}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'csv' or 'json'.")
    
    def export_prs_to_csv(self, prs: List[Dict[str, Any]], filename: str) -> None:
        """
        Legacy method for backward compatibility.
        Export PR data to a CSV file including detailed file change information.
        
        Args:
            prs (List[Dict[str, Any]]): List of PRs to export.
            filename (str): The filename for the CSV output.
        """
        pr_details = self.collect_pr_details(prs)
        self.export_pr_details(pr_details, filename, format='csv')
    
    @staticmethod
    def analyze_merged_pr_stats(json_file: str) -> pd.DataFrame:
        """
        Analyze merged PRs from exported JSON data, focusing on source code files and statistics.
        
        Args:
            json_file (str): Path to the exported JSON file containing PR details.
            
        Returns:
            pd.DataFrame: Statistics of merged PRs including file counts and line changes.
        """
        # Read JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            pr_details = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(pr_details)
        
        # Filter for merged PRs
        df_merged = df[df['PR State'] == 'merged']
        # df_merged = df
        
        # Define source code file extensions
        source_extensions = {'.py', '.cpp', '.c', '.h', '.hpp', '.java', '.js', '.ts', '.sh', '.yaml', '.yml'}
        
        # Filter for source code files
        is_source = df_merged['File Name'].apply(
            lambda x: any(x.endswith(ext) for ext in source_extensions) if isinstance(x, str) else False
        )
        df_source = df_merged[is_source]
        
        # Group by PR and calculate statistics
        pr_stats = df_source.groupby('PR Number').agg({
            'PR Title': 'first',
            'Created At': 'first',
            'Merged At': 'first',
            'File Name': 'count',
            'Added Lines': 'sum',
            'Removed Lines': 'sum',
            'Total Changed Lines': 'sum'
        }).rename(columns={
            'File Name': 'Source Files Changed',
            'Added Lines': 'Total Lines Added',
            'Removed Lines': 'Total Lines Removed',
            'Total Changed Lines': 'Total Lines Changed'
        })
        
        # Calculate time to merge
        pr_stats['Created At'] = pd.to_datetime(pr_stats['Created At'])
        pr_stats['Merged At'] = pd.to_datetime(pr_stats['Merged At'])
        pr_stats['Days to Merge'] = (pr_stats['Merged At'] - pr_stats['Created At']).dt.total_seconds() / (24 * 3600)
        
        # Add summary statistics
        total_stats = {
            'Total Merged PRs': len(pr_stats),
            'Total Source Files Changed': pr_stats['Source Files Changed'].sum(),
            'Total Lines Added': pr_stats['Total Lines Added'].sum(),
            'Total Lines Removed': pr_stats['Total Lines Removed'].sum(),
            'Total Lines Changed': pr_stats['Total Lines Changed'].sum(),
            'Average Days to Merge': pr_stats['Days to Merge'].mean(),
            'Median Days to Merge': pr_stats['Days to Merge'].median(),
        }
        
        return pr_stats, total_stats

def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and analyze Gitee Pull Requests')
    parser.add_argument('--owner', required=True, help='Repository owner/organization')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--author', required=True, help='PR author username')
    parser.add_argument('--state', default='all', choices=['all', 'open', 'closed', 'merged'],
                      help='PR state to include (default: all)')
    parser.add_argument('--per_page', type=int, default=50, help='Number of PRs per page (default: 50)')
    parser.add_argument('--max_pages', type=int, default=5, help='Maximum number of pages to fetch (default: 5)')
    parser.add_argument('--since', type=lambda s: datetime.fromisoformat(s),
                      help='Only include PRs after this date (ISO format: YYYY-MM-DD)')
    parser.add_argument('--until', type=lambda s: datetime.fromisoformat(s),
                      help='Only include PRs before this date (ISO format: YYYY-MM-DD)')
    parser.add_argument('--output_dir', default='output', help='Output directory for exported files')
    
    return parser.parse_args()

from py3_tools.py_debug.debug_utils import Debugger
# Debugger.debug_flag = True  # Enable debug mode globally
@Debugger.attach_on_error()
def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to file in output directory
    log_file = output_dir / f"gitee_pr_stat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    config = GiteePRConfig(
        owner=args.owner,
        repo=args.repo,
        author_name=args.author,
        state=args.state,
        per_page=args.per_page,
        max_pages=args.max_pages,
        since=args.since,
        until=args.until,
        output_dir=str(output_dir)
    )
    
    client = GiteePRClient(config)
    prs = client.fetch_prs()
    
    # Display basic PR information
    for pr in prs:
        print(f"#{pr['number']}: {pr['title']} ({pr['html_url']})")
    print(f"Fetched {len(prs)} PRs from {config.owner}/{config.repo} for user '{config.author_name}' (ID: {client.author_id})")
    
    # Collect PR details
    pr_details = client.collect_pr_details(prs)
    print(f"Collected details for {len(pr_details)} PR file changes")
    
    # Export files to output directory
    base_filename = f"gitee_prs_{config.owner}_{config.repo}_{config.author_name}"
    
    # Export to CSV
    csv_path = output_dir / f"{base_filename}.csv"
    client.export_pr_details(pr_details, str(csv_path), format='csv')
    print(f"Exported PR details to CSV: {csv_path}")
    
    # Export to JSON
    json_path = output_dir / f"{base_filename}.json"
    client.export_pr_details(pr_details, str(json_path), format='json')
    print(f"Exported PR details to JSON: {json_path}")
    
    # Analyze merged PR statistics
    print("\nAnalyzing merged PR statistics...")
    pr_stats, total_stats = client.analyze_merged_pr_stats(str(json_path))
    
    # Display summary statistics
    print("\nSummary Statistics:")
    for key, value in total_stats.items():
        if 'Days' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Export PR statistics
    stats_base = output_dir / f"{base_filename}_stats"
    pr_stats.to_csv(f"{stats_base}_details.csv")
    pd.DataFrame([total_stats]).to_csv(f"{stats_base}_summary.csv")
    print(f"\nExported PR statistics to CSV files:")
    print(f"PR Statistics: {stats_base}_details.csv")
    print(f"Summary: {stats_base}_summary.csv")

if __name__ == '__main__':
    main()