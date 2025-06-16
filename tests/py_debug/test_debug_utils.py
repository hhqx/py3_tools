"""Unit tests for debug utilities."""
import unittest
from unittest import mock
import os
import sys
import io

from src.py_debug.debug_utils import (
    RemotePdb,
    WebPdb, 
    DistributedDebugger,
    setup_debugger,
    DummyDebugger
)


class TestDistributedDebugger(unittest.TestCase):
    """Test cases for DistributedDebugger."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_is_master_with_rank(self):
        """Test is_master with RANK environment variable."""
        # Test master process
        os.environ['RANK'] = '0'
        self.assertTrue(DistributedDebugger.is_master())
        
        # Test worker process
        os.environ['RANK'] = '1'
        self.assertFalse(DistributedDebugger.is_master())
    
    def test_is_master_with_slurm(self):
        """Test is_master with SLURM environment variable."""
        # Test master process
        os.environ['SLURM_PROCID'] = '0'
        self.assertTrue(DistributedDebugger.is_master())
        
        # Test worker process
        os.environ['SLURM_PROCID'] = '1'
        self.assertFalse(DistributedDebugger.is_master())
    
    def test_get_rank(self):
        """Test get_rank method."""
        # Test with RANK
        os.environ['RANK'] = '3'
        self.assertEqual(DistributedDebugger.get_rank(), 3)
        
        # Test with SLURM
        del os.environ['RANK']
        os.environ['SLURM_PROCID'] = '2'
        self.assertEqual(DistributedDebugger.get_rank(), 2)
        
        # Test default
        del os.environ['SLURM_PROCID']
        self.assertEqual(DistributedDebugger.get_rank(), 0)


class TestSetupDebugger(unittest.TestCase):
    """Test cases for setup_debugger function."""
    
    @mock.patch('src.py_debug.debug_utils.RemotePdb')
    def test_setup_remote_debugger(self, mock_remote_pdb):
        """Test setting up remote debugger."""
        debugger = setup_debugger(debugger_type='remote', host='localhost', port=1234)
        mock_remote_pdb.assert_called_once_with(host='localhost', port=1234)
    
    @mock.patch('src.py_debug.debug_utils.WebPdb')
    def test_setup_web_debugger(self, mock_web_pdb):
        """Test setting up web debugger."""
        debugger = setup_debugger(debugger_type='web', host='localhost', port=5678)
        mock_web_pdb.assert_called_once_with(host='localhost', port=5678)
    
    def test_setup_unknown_debugger(self):
        """Test setting up unknown debugger type."""
        with self.assertRaises(ValueError):
            setup_debugger(debugger_type='invalid')
    
    @mock.patch('src.py_debug.debug_utils.DistributedDebugger')
    def test_master_only_filtering(self, mock_dist_debugger):
        """Test master-only filtering."""
        # Set up worker process
        mock_dist_debugger.get_rank.return_value = 1
        
        # Should get dummy debugger for worker
        debugger = setup_debugger(only_master=True)
        self.assertIsInstance(debugger, DummyDebugger)
        
        # Set up master process
        mock_dist_debugger.get_rank.return_value = 0
        
        # Should get real debugger for master
        with mock.patch('src.py_debug.debug_utils.RemotePdb') as mock_pdb:
            debugger = setup_debugger(only_master=True)
            mock_pdb.assert_called_once()


if __name__ == '__main__':
    unittest.main()
