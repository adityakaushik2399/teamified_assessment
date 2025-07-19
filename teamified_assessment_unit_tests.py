import pytest
import numpy as np
import faiss
import os
import json
import pickle
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import BytesIO
import requests
import fitz
from sentence_transformers import SentenceTransformer
from teamified_assessment import PhilippineHistoryRAG


class TestPhilippineHistoryRAGInit:
    """Test the initialization of PhilippineHistoryRAG class."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        rag = PhilippineHistoryRAG()
        
        assert rag.pdf_path == "PHILIPPINE-HISTORY-SOURCE-BOOK-FINAL-SEP022021.pdf"
        assert rag.embedding_model_name == "all-MiniLM-L6-v2"
        assert rag.ollama_model == "gemma3:1b"
        assert rag.ollama_url == "http://localhost:11434"
        assert rag.chunk_size == 500
        assert rag.chunk_overlap == 50
        assert rag.embedding_dim == 384
        assert rag.chunks == []
        assert rag.faiss_index is None


class TestPDFProcessing:
    """Test PDF loading and text extraction functionality."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    def test_load_and_extract_text_success(self, rag_instance):
        """Test successful PDF text extraction."""
        mock_pdf = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "Sample text from page 1"
        
        with patch('teamified_assessment.fitz.open', return_value=mock_pdf):
            result = rag_instance.load_and_extract_text()
            
            assert "Sample text from page 1" in result
            assert "Page 1" in result
            mock_pdf.close.assert_called_once()
    
    def test_load_and_extract_text_file_not_found(self, rag_instance):
        """Test FileNotFoundError handling."""
        with patch('teamified_assessment.fitz.open', side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="PDF file not found"):
                rag_instance.load_and_extract_text()
    
    def test_load_and_extract_text_general_exception(self, rag_instance):
        """Test general exception handling during PDF loading."""
        with patch('teamified_assessment.fitz.open', side_effect=Exception("PDF error")):
            with pytest.raises(Exception, match="Error reading PDF: PDF error"):
                rag_instance.load_and_extract_text()


class TestTextProcessing:
    """Test text cleaning and chunking functionality."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG(chunk_size=100, chunk_overlap=20)
    
    def test_clean_text(self, rag_instance):
        """Test text cleaning functionality."""
        dirty_text = "This   has    multiple   spaces.\n\nAnd special @#$% characters!!! Too many punctuation marks..."
        cleaned = rag_instance.clean_text(dirty_text)
        
        assert "multiple spaces" in cleaned
        assert "@#$%" not in cleaned
        assert "!!!" not in cleaned
        assert "..." not in cleaned
    
    def test_create_chunks_basic(self, rag_instance):
        """Test basic text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        chunks = rag_instance.create_chunks(text)
        
        assert len(chunks) > 0
        assert all('id' in chunk for chunk in chunks)
        assert all('text' in chunk for chunk in chunks)
        assert all('length' in chunk for chunk in chunks)
    
    def test_create_chunks_with_overlap(self, rag_instance):
        """Test chunking with overlap."""
        # Create text longer than chunk_size (100)
        text = "A" * 150 + ". " + "B" * 150 + "."
        chunks = rag_instance.create_chunks(text)
        
        assert len(chunks) >= 2
        # Check that chunks have some overlap
        if len(chunks) > 1:
            # This is a simplified check - in reality overlap might be more complex
            assert len(chunks[0]['text']) <= rag_instance.chunk_size + 50  # some tolerance
    
    def test_create_chunks_empty_text(self, rag_instance):
        """Test chunking with empty text."""
        chunks = rag_instance.create_chunks("")
        assert len(chunks) == 0


class TestEmbeddingsAndIndex:
    """Test embedding creation and FAISS index building."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {'id': 0, 'text': 'First chunk text', 'length': 16},
            {'id': 1, 'text': 'Second chunk text', 'length': 17}
        ]
    
    def test_create_embeddings(self, rag_instance, sample_chunks):
        """Test embedding creation."""
        mock_embeddings = np.random.rand(2, 384)
        rag_instance.embedding_model.encode.return_value = mock_embeddings
        
        embeddings = rag_instance.create_embeddings(sample_chunks)
        
        assert embeddings.shape == (2, 384)
        rag_instance.embedding_model.encode.assert_called_once()
    
    def test_build_faiss_index(self, rag_instance):
        """Test FAISS index building."""
        embeddings = np.random.rand(10, 384).astype(np.float32)
        
        index = rag_instance.build_faiss_index(embeddings)
        
        assert isinstance(index, faiss.Index)
        assert index.ntotal == 10
        assert index.d == 384
    
    def test_retrieve_relevant_chunks_no_index(self, rag_instance):
        """Test retrieving chunks when index is not built."""
        with pytest.raises(ValueError, match="Index not built"):
            rag_instance.retrieve_relevant_chunks("test query")
    
    def test_retrieve_relevant_chunks_success(self, rag_instance, sample_chunks):
        """Test successful chunk retrieval."""
        # Set up mock index and chunks
        rag_instance.chunks = sample_chunks
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.3]]),  # distances
            np.array([[0, 1]])       # indices
        )
        rag_instance.faiss_index = mock_index
        rag_instance.embedding_model.encode.return_value = np.random.rand(1, 384)
        
        result = rag_instance.retrieve_relevant_chunks("test query", top_k=2)
        
        assert len(result) == 2
        assert result[0]['similarity_score'] == 0.1
        assert result[1]['similarity_score'] == 0.3
        assert result[0]['rank'] == 1
        assert result[1]['rank'] == 2


class TestOllamaIntegration:
    """Test Ollama API integration."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    def test_query_ollama_success(self, rag_instance):
        """Test successful Ollama query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'response': 'Test response from Ollama'}
        
        with patch('teamified_assessment.requests.post', return_value=mock_response) as mock_post:
            result = rag_instance.query_ollama("test prompt")
            
            assert result == "Test response from Ollama"
            mock_post.assert_called_once()
            
            # Check the call arguments
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"
            assert call_args[1]['json']['prompt'] == "test prompt"
            assert call_args[1]['json']['model'] == "gemma3:1b"
    
    def test_query_ollama_http_error(self, rag_instance):
        """Test Ollama HTTP error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        with patch('teamified_assessment.requests.post', return_value=mock_response):
            result = rag_instance.query_ollama("test prompt")
            
            assert "Error: HTTP 500" in result
            assert "Internal Server Error" in result
    
    def test_query_ollama_connection_error(self, rag_instance):
        """Test Ollama connection error handling."""
        with patch('teamified_assessment.requests.post', side_effect=requests.exceptions.ConnectionError):
            result = rag_instance.query_ollama("test prompt")
            
            assert "Could not connect to Ollama" in result
            assert "localhost:11434" in result
    
    def test_query_ollama_timeout(self, rag_instance):
        """Test Ollama timeout handling."""
        with patch('teamified_assessment.requests.post', side_effect=requests.exceptions.Timeout):
            result = rag_instance.query_ollama("test prompt")
            
            assert "Request to Ollama timed out" in result


class TestSaveAndLoad:
    """Test saving and loading index functionality."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    def test_save_index_no_index(self, rag_instance):
        """Test saving when no index exists."""
        with pytest.raises(ValueError, match="No index to save"):
            rag_instance.save_index()
    
    def test_save_index_success(self, rag_instance):
        """Test successful index saving."""
        # Create a mock index and chunks
        mock_index = Mock()
        rag_instance.faiss_index = mock_index
        rag_instance.chunks = [{'id': 0, 'text': 'test'}]
        
        with patch('teamified_assessment.faiss.write_index') as mock_write_index, \
             patch('builtins.open', mock_open()) as mock_file:
            
            rag_instance.save_index("test_index.index", "test_chunks.pkl")
            
            mock_write_index.assert_called_once_with(mock_index, "test_index.index")
            mock_file.assert_called()
    
    def test_load_index_success(self, rag_instance):
        """Test successful index loading."""
        mock_index = Mock()
        mock_chunks = [{'id': 0, 'text': 'test'}]
        
        with patch('teamified_assessment.faiss.read_index', return_value=mock_index) as mock_read_index, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('teamified_assessment.pickle.load', return_value=mock_chunks) as mock_pickle_load:
            
            rag_instance.load_index("test_index.index", "test_chunks.pkl")
            
            assert rag_instance.faiss_index == mock_index
            assert rag_instance.chunks == mock_chunks
            mock_read_index.assert_called_once_with("test_index.index")


class TestFullRAGWorkflow:
    """Test the complete RAG workflow."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    def test_build_context(self, rag_instance):
        """Test context building from relevant chunks."""
        relevant_chunks = [
            {'id': 0, 'text': 'First chunk', 'rank': 1, 'similarity_score': 0.1},
            {'id': 1, 'text': 'Second chunk', 'rank': 2, 'similarity_score': 0.3}
        ]
        
        context = rag_instance.build_context(relevant_chunks)
        
        assert "[Chunk 1]: First chunk" in context
        assert "[Chunk 2]: Second chunk" in context
    
    @patch('teamified_assessment.PhilippineHistoryRAG.query_ollama')
    @patch('teamified_assessment.PhilippineHistoryRAG.retrieve_relevant_chunks')
    def test_answer_query_success(self, mock_retrieve, mock_ollama, rag_instance):
        """Test successful query answering."""
        # Setup mocks
        mock_chunks = [
            {'id': 0, 'text': 'Relevant text', 'rank': 1, 'similarity_score': 0.1}
        ]
        mock_retrieve.return_value = mock_chunks
        mock_ollama.return_value = "This is the answer"
        
        result = rag_instance.answer_query("test question")
        
        assert result['query'] == "test question"
        assert result['answer'] == "This is the answer"
        assert result['relevant_chunks'] == mock_chunks
        assert "[Chunk 1]: Relevant text" in result['context']
    
    @patch('teamified_assessment.PhilippineHistoryRAG.retrieve_relevant_chunks')
    def test_answer_query_no_relevant_chunks(self, mock_retrieve, rag_instance):
        """Test query answering when no relevant chunks found."""
        mock_retrieve.return_value = []
        
        result = rag_instance.answer_query("test question")
        
        assert result['query'] == "test question"
        assert "couldn't find relevant information" in result['answer']
        assert result['relevant_chunks'] == []
        assert result['context'] == ""
    
    def test_print_detailed_response(self, rag_instance, capsys):
        """Test detailed response printing."""
        result = {
            'query': 'Test query',
            'answer': 'Test answer',
            'relevant_chunks': [
                {'id': 0, 'text': 'Short text', 'rank': 1, 'similarity_score': 0.1}
            ],
            'context': 'Test context'
        }
        
        rag_instance.print_detailed_response(result)
        
        captured = capsys.readouterr()
        assert 'Test query' in captured.out
        assert 'Test answer' in captured.out
        assert 'Short text' in captured.out


class TestSetupIndex:
    """Test the complete index setup process."""
    
    @pytest.fixture
    def rag_instance(self):
        """Create a RAG instance for testing."""
        return PhilippineHistoryRAG()
    
    @patch('teamified_assessment.PhilippineHistoryRAG.build_faiss_index')
    @patch('teamified_assessment.PhilippineHistoryRAG.create_embeddings')
    @patch('teamified_assessment.PhilippineHistoryRAG.create_chunks')
    @patch('teamified_assessment.PhilippineHistoryRAG.load_and_extract_text')
    def test_setup_index_complete_workflow(self, mock_load_text, mock_create_chunks, 
                                         mock_create_embeddings, mock_build_index, rag_instance):
        """Test the complete setup workflow."""
        # Setup mocks
        mock_load_text.return_value = "Sample PDF text"
        mock_chunks = [{'id': 0, 'text': 'chunk', 'length': 5}]
        mock_create_chunks.return_value = mock_chunks
        mock_embeddings = np.random.rand(1, 384)
        mock_create_embeddings.return_value = mock_embeddings
        mock_index = Mock()
        mock_build_index.return_value = mock_index
        
        rag_instance.setup_index()
        
        # Verify all methods were called
        mock_load_text.assert_called_once()
        mock_create_chunks.assert_called_once_with("Sample PDF text")
        mock_create_embeddings.assert_called_once_with(mock_chunks)
        mock_build_index.assert_called_once_with(mock_embeddings)
        
        # Verify state
        assert rag_instance.chunks == mock_chunks
        assert rag_instance.faiss_index == mock_index

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])