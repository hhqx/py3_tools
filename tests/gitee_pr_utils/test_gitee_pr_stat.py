"""Tests for Gitee PR statistics utilities."""
import json
import os
import unittest
from datetime import datetime
from unittest import mock

import requests

from py3_tools.gitee.gitee_pr_stat import GiteePRConfig, GiteePRClient


class TestGiteePRClient(unittest.TestCase):
    """Test cases for GiteePRClient."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = GiteePRConfig(
            owner="ascend",
            repo="msit",
            author_name="hhhqx",
            state="all",
            per_page=50,
            max_pages=3,
            since=datetime(2025, 5, 1),
            until=datetime(2025, 6, 13),
        )

        # Load sample data for mocking
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "src", "gitee_pr_utils")
        
        # User info fixture
        user_file = os.path.join(src_dir, "hhhqx")
        with open(user_file, "r") as f:
            self.user_data = json.load(f)
            
        # PR list fixture
        pr_file = os.path.join(src_dir, "o.json")
        with open(pr_file, "r") as f:
            self.pr_data = json.load(f)

    @mock.patch("requests.Session")
    def test_resolve_author_id(self, mock_session):
        """Test resolving author ID from username."""
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.user_data
        
        mock_session_instance = mock_session.return_value
        mock_session_instance.get.return_value = mock_response
        
        client = GiteePRClient(self.test_config)
        self.assertEqual(client.author_id, self.user_data["id"])
        
        # Check API call
        mock_session_instance.get.assert_called_once()
        call_args = mock_session_instance.get.call_args[0][0]
        self.assertIn("users/hhhqx", call_args)
        
    @mock.patch("requests.Session")
    def test_fetch_prs(self, mock_session):
        """Test fetching PRs with filtering."""
        # Mock user response
        user_response = mock.Mock()
        user_response.status_code = 200
        user_response.json.return_value = self.user_data
        
        # Mock PR response
        pr_response = mock.Mock()
        pr_response.status_code = 200
        pr_response.json.return_value = self.pr_data
        
        mock_session_instance = mock_session.return_value
        mock_session_instance.get.side_effect = [user_response, pr_response]
        
        client = GiteePRClient(self.test_config)
        prs = client.fetch_prs()
        
        # Verify API calls
        self.assertEqual(mock_session_instance.get.call_count, 2)
        
        # Verify PR filtering logic
        for pr in prs:
            # Ensure all PRs are from the correct author
            self.assertEqual(pr["user"]["id"], client.author_id)
            
            # Ensure time filtering
            pr_time_str = pr.get("updated_at") or pr.get("created_at")
            pr_time = datetime.fromisoformat(pr_time_str.rstrip("Z"))
            self.assertTrue(self.test_config.since <= pr_time <= self.test_config.until)
            
    @mock.patch("requests.Session")
    def test_user_not_found(self, mock_session):
        """Test behavior when user is not found."""
        mock_response = mock.Mock()
        mock_response.status_code = 404
        
        mock_session_instance = mock_session.return_value
        mock_session_instance.get.return_value = mock_response
        
        config = GiteePRConfig(
            owner="ascend",
            repo="msit",
            author_name="nonexistent_user",
            state="all"
        )
        
        with self.assertRaises(ValueError):
            GiteePRClient(config)


if __name__ == "__main__":
    unittest.main()
