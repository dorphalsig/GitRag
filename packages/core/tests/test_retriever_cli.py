import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from Chunker.Chunk import Chunk
from Retriever import main as retriever_main


def test_retriever_cli_search():
    mock_chunk = Chunk(
        chunk="test chunk",
        repo="test/repo",
        path="test.py",
        language="python",
        start_rc=(1, 0),
        end_rc=(2, 0),
        start_bytes=0,
        end_bytes=10,
    )

    with patch("Retriever.load_components") as mock_load, \
         patch("Retriever.Retriever") as mock_retriever_class, \
         patch("sys.argv", ["Retriever.py", "test query", "--top-k", "5"]), \
         patch("sys.stdout") as mock_stdout:

        mock_calc = MagicMock()
        mock_persist = MagicMock()
        mock_load.return_value = (mock_calc, mock_persist)

        mock_retriever = mock_retriever_class.return_value
        mock_retriever.retrieve.return_value = [mock_chunk]

        retriever_main()

        mock_retriever.retrieve.assert_called_once_with("test query", top_k=5, repo=None, branch=None)

        # Verify output is JSON
        output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["chunk"] == "test chunk"


def test_retriever_cli_server():
    with patch("Retriever.load_components") as mock_load, \
         patch("Retriever.Retriever"), \
         patch("Retriever.Qwen3Reranker"), \
         patch("gitrag_mcp_server.server.create_mcp_server") as mock_create_server, \
         patch("sys.argv", ["Retriever.py", "--server", "--port", "8000"]):

        mock_load.return_value = (MagicMock(), MagicMock())
        mock_server = mock_create_server.return_value

        retriever_main()

        mock_create_server.assert_called_once()
        mock_server.run.assert_called_once_with(transport="sse", port=8000, host="0.0.0.0")


def test_retriever_cli_no_query():
    with patch("sys.argv", ["Retriever.py"]), \
         patch("sys.exit") as mock_exit, \
         patch("Retriever.load_components") as mock_load:
        
        mock_exit.side_effect = SystemExit(1)
        try:
            retriever_main()
        except SystemExit:
            pass
            
        mock_exit.assert_called_once_with(1)
        mock_load.assert_not_called()
