# teamified_assessment
Assessment for the teamified task

Python version for the task : Python3.13

Steps to run the Code:
  - Create a virtual environment
  - `pip install -r requirements.txt`
  - `python3 teamified_assessment.py`

LLM Used:
  - Used Ollama to run the model
  - Used Google's `gemma3:1b` model as my LLM choice

Used this LLM only because this is a very lightweight and highly optimized model for these tasks. Also, for future requirements, it can be fine-tuned very easily to further reduce inference times.

Demo run of the code:

```
(teamified_assessment) aditya@aditya-Dell-G15-5520:~/teamified_assessment$ python3 teamified_assessment.py 
Loading embedding model...
Found existing index files. Loading...
Index loaded from faiss_index.index, chunks loaded from chunks.pkl
Philippine History RAG System is ready!

Sample queries you can try:
1. When did the EDSA People Power Revolution happen?
2. Who was Jose Rizal?
3. What happened during the Spanish colonization of the Philippines?
4. Tell me about Ferdinand Marcos and Martial Law
5. What was the Katipunan?

==================================================
Enter your question about Philippine history (or 'quit' to exit): What happened during the Spanish colonization of the Philippines?
Processing query: What happened during the Spanish colonization of the Philippines?
--------------------------------------------------
Retrieving answer from Ollama...
============================================================
PHILIPPINE HISTORY RAG SYSTEM - QUERY RESULT
============================================================
Query: What happened during the Spanish colonization of the Philippines?
------------------------------------------------------------
ANSWER:
The Spanish colonization of the Philippines ended in 1898, and it began with the set of materials presenting the various facets of colonial life during the nineteenth century.
------------------------------------------------------------
RETRIEVED CONTEXT (3 chunks):

[Chunk 1 - Score: 0.5521]
ending the Spanish control of the islands in 1898. The set of materials included here presents the various facets of colonial life in the nineteenth century Philippines. These were written by various ...

[Chunk 2 - Score: 0.6753]
come firmer and stronger with the passing of time. The history of the Filipino people is an epic of an unrelenting struggle for freedom. Centuries ago, our forebears sailed in frail vintas to settle i...

[Chunk 3 - Score: 0.6971]
d the Arts. 2001 Hernandez, Jose Rhommel B. trans. An Ethnographic Description of the Peoples of Luzon by Fr. Juan Ferrando, O.P. in Colloquia Manilana, Vol. 2 (1994): 85-100  Insurrections by Filipin...
============================================================

==================================================
Enter your question about Philippine history (or 'quit' to exit): 
Please enter a valid question.

==================================================
Enter your question about Philippine history (or 'quit' to exit): What was the Katipunan?
Processing query: What was the Katipunan?
--------------------------------------------------
Retrieving answer from Ollama...
============================================================
PHILIPPINE HISTORY RAG SYSTEM - QUERY RESULT
============================================================
Query: What was the Katipunan?
------------------------------------------------------------
ANSWER:
The Katipunan was an organization focused on internal affairs, particularly the imprisonment of a friar and the copying of the Himno Nacional. It aimed to promote noble and clean living and was guided by the Kartilya.
------------------------------------------------------------
RETRIEVED CONTEXT (3 chunks):

[Chunk 1 - Score: 0.7273]
glimpse to the internal affairs of the Katipunan. Two interesting matter arose from the document. The first one is one the subject of an imprisoned Augustinian friar by the name of Antonio Piernavieja...

[Chunk 2 - Score: 0.7728]
authority of the sacred commands of the Katipunan. All acts contrary to noble and clean living are repugnant here, and hence the life of anyone who wants to affiliate with this Association will be sub...

[Chunk 3 - Score: 0.8055]
is nom de guerre) and  Dimasilaw  (his pseudonym). According to the historian Jim Richardson, the  Kartilya  served as the guiding principles and primary teachings of the Katipunan. The contents of th...
============================================================

==================================================
Enter your question about Philippine history (or 'quit' to exit): quit
Thank you for using the Philippine History RAG system!
(teamified_assessment) aditya@aditya-Dell-G15-5520:~/teamified_assessment$ 

```

UNIT TESTS FOR THE CODE

```
(teamified_assessment) aditya@aditya-Dell-G15-5520:~/teamified_assessment$ pytest teamified_assessment_unit_tests.py -v
==================================================================================== test session starts =====================================================================================
platform linux -- Python 3.13.5, pytest-8.4.1, pluggy-1.6.0 -- /home/aditya/teamified_assessment/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aditya/teamified_assessment
plugins: mock-3.14.1, cov-6.2.1, langsmith-0.4.8, anyio-4.9.0
collected 24 items                                                                                                                                                                           

teamified_assessment_unit_tests.py::TestPhilippineHistoryRAGInit::test_init_default_parameters PASSED                                                                                  [  4%]
teamified_assessment_unit_tests.py::TestPDFProcessing::test_load_and_extract_text_success FAILED                                                                                       [  8%]
teamified_assessment_unit_tests.py::TestPDFProcessing::test_load_and_extract_text_file_not_found PASSED                                                                                [ 12%]
teamified_assessment_unit_tests.py::TestPDFProcessing::test_load_and_extract_text_general_exception PASSED                                                                             [ 16%]
teamified_assessment_unit_tests.py::TestTextProcessing::test_clean_text PASSED                                                                                                         [ 20%]
teamified_assessment_unit_tests.py::TestTextProcessing::test_create_chunks_basic PASSED                                                                                                [ 25%]
teamified_assessment_unit_tests.py::TestTextProcessing::test_create_chunks_with_overlap FAILED                                                                                         [ 29%]
teamified_assessment_unit_tests.py::TestTextProcessing::test_create_chunks_empty_text PASSED                                                                                           [ 33%]
teamified_assessment_unit_tests.py::TestEmbeddingsAndIndex::test_create_embeddings FAILED                                                                                              [ 37%]
teamified_assessment_unit_tests.py::TestEmbeddingsAndIndex::test_build_faiss_index PASSED                                                                                              [ 41%]
teamified_assessment_unit_tests.py::TestEmbeddingsAndIndex::test_retrieve_relevant_chunks_no_index PASSED                                                                              [ 45%]
teamified_assessment_unit_tests.py::TestEmbeddingsAndIndex::test_retrieve_relevant_chunks_success FAILED                                                                               [ 50%]
teamified_assessment_unit_tests.py::TestOllamaIntegration::test_query_ollama_success PASSED                                                                                            [ 54%]
teamified_assessment_unit_tests.py::TestOllamaIntegration::test_query_ollama_http_error PASSED                                                                                         [ 58%]
teamified_assessment_unit_tests.py::TestOllamaIntegration::test_query_ollama_connection_error PASSED                                                                                   [ 62%]
teamified_assessment_unit_tests.py::TestOllamaIntegration::test_query_ollama_timeout PASSED                                                                                            [ 66%]
teamified_assessment_unit_tests.py::TestSaveAndLoad::test_save_index_no_index PASSED                                                                                                   [ 70%]
teamified_assessment_unit_tests.py::TestSaveAndLoad::test_save_index_success PASSED                                                                                                    [ 75%]
teamified_assessment_unit_tests.py::TestSaveAndLoad::test_load_index_success PASSED                                                                                                    [ 79%]
teamified_assessment_unit_tests.py::TestFullRAGWorkflow::test_build_context PASSED                                                                                                     [ 83%]
teamified_assessment_unit_tests.py::TestFullRAGWorkflow::test_answer_query_success PASSED                                                                                              [ 87%]
teamified_assessment_unit_tests.py::TestFullRAGWorkflow::test_answer_query_no_relevant_chunks PASSED                                                                                   [ 91%]
teamified_assessment_unit_tests.py::TestFullRAGWorkflow::test_print_detailed_response PASSED                                                                                           [ 95%]
teamified_assessment_unit_tests.py::TestSetupIndex::test_setup_index_complete_workflow PASSED                                                                                          [100%]

```
