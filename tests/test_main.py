# import pytest
# import os
# from src.main import get_index_name, validate_vector_store_config

# def test_get_index_name_openai(monkeypatch):
#     """Test getting OpenAI index name"""
#     # Mock environment variable
#     monkeypatch.setenv('OPENAI_INDEX_NAME', 'test-openai-index')
    
#     # Test
#     assert get_index_name('OpenAI') == 'test-openai-index'

# def test_get_index_name_llama(monkeypatch):
#     """Test getting Llama index name"""
#     # Mock environment variable
#     monkeypatch.setenv('LLAMA_3_INDEX_NAME', 'test-llama-index')
    
#     # Test
#     assert get_index_name('Llama 3') == 'test-llama-index'

# def test_get_index_name_missing():
#     """Test error when index name is missing"""
#     with pytest.raises(ValueError) as exc_info:
#         get_index_name('OpenAI')
#     assert "Please set OPENAI_INDEX_NAME in your .env file" in str(exc_info.value)

# def test_validate_config_missing_pinecone():
#     """Test error when Pinecone API key is missing"""
#     with pytest.raises(ValueError) as exc_info:
#         validate_vector_store_config()
#     assert "PINECONE_API_KEY not found" in str(exc_info.value)

# def test_validate_config_missing_openai():
#     """Test error when OpenAI API key is missing"""
#     # Mock Pinecone key but not OpenAI
#     os.environ['PINECONE_API_KEY'] = 'dummy-pinecone-key'
#     with pytest.raises(ValueError) as exc_info:
#         validate_vector_store_config()
#     assert "OPENAI_API_KEY not found" in str(exc_info.value)

# def test_validate_config_success(monkeypatch):
#     """Test successful configuration validation"""
#     # Mock both required keys
#     monkeypatch.setenv('PINECONE_API_KEY', 'dummy-pinecone-key')
#     monkeypatch.setenv('OPENAI_API_KEY', 'dummy-openai-key')
    
#     # Should not raise any exceptions
#     validate_vector_store_config() 