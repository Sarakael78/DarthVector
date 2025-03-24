import pytest
import tempfile
from pathlib import Path
import numpy as np
from rtfProcessor import extractTextFromRtf, RTFEncodingError, FileSizeExceededError
from textPreprocessor import TextPreprocessor
from vectorStore import VectorStore
from main import RtfProcessingPipeline
from config import config


@pytest.fixture
def tempRtfFile():
    with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False) as f:
        f.write(b"{\\rtf1\\ansi This is a test}")
    file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)


@pytest.fixture
def tempLargeRtfFile():
    with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False) as f:
        f.write(b"{\\rtf1\\ansi " + b"A" * int(0.1 * 1024 * 1024) + b"}")
    file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)


@pytest.fixture
def tempNonRtfFile():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"This is not an RTF file.")
    file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)


@pytest.fixture
def tempRtfFileWithEncoding():
    content = "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Arial;}}This is a test with some special characters: éàçüö}"
    with tempfile.NamedTemporaryFile(suffix=".rtf", delete=False, mode="w", encoding="latin-1") as f:
        f.write(content)
    file_path = Path(f.name)
    yield file_path
    file_path.unlink(missing_ok=True)


class TestTextPreprocessor:
    def testCleanText(self):
        text = "This is\n\na test   with\tspaces"
        assert TextPreprocessor.cleanText(text) == "This is a test with spaces"

    def testSegmentText(self):
        preprocessor = TextPreprocessor(maxWords=5, chunkOverlap=2)
        text = "This is a test of the segmentation function with overlapping words"
        segments = preprocessor.segmentText(text)
        assert len(segments) == 3
        assert segments[0] == "This is a test of"
        assert segments[1] == "of the segmentation function with"
        assert "with overlapping words" in segments[2]

    def testSegmentTextEmpty(self):
        preprocessor = TextPreprocessor(maxWords=5, chunkOverlap=2)
        text = ""
        segments = preprocessor.segmentText(text)
        assert segments == []


class TestVectorStore:
    def testAddSearch(self):
        embeddingDimension = 3
        store = VectorStore(embeddingDimension=embeddingDimension)
        embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        metadata = [{"id": i} for i in range(3)]
        store.addEmbeddings(embeddings, metadata)
        assert store.index.ntotal == 3
        queryEmbedding = np.array([1, 1, 1], dtype=np.float32)
        distances, _, results = store.search(queryEmbedding, topK=2)
        assert len(results) == 2
        assert results[0]["id"] == 0

    def testSaveLoad(self):
        embeddingDimension = 3
        store = VectorStore(embeddingDimension=embeddingDimension)
        embeddings = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        metadata = [{"id": 1}, {"id": 2}]
        store.addEmbeddings(embeddings, metadata)

        with tempfile.TemporaryDirectory() as tempDir:
            indexPath = Path(tempDir) / "test_index.faiss"
            metadataPath = Path(tempDir) / "test_metadata.pkl"
            store.save(indexPath=str(indexPath), metadataPath=str(metadataPath))

            loadedStore = VectorStore(embeddingDimension=embeddingDimension)
            loadedStore.load(indexPath=str(indexPath), metadataPath=str(metadataPath))

            assert loadedStore.index.ntotal == 2
            assert len(loadedStore.metadata) == 2
            assert loadedStore.metadata[0]["id"] == 1

    def testSearchEmpty(self):
        embeddingDimension = 3
        store = VectorStore(embeddingDimension=embeddingDimension)
        queryEmbedding = np.array([1, 2, 3], dtype=np.float32)
        distances, indices, metadata = store.search(queryEmbedding, topK=1)
        assert len(distances) == 0
        assert len(indices) == 0
        assert len(metadata) == 0


class TestRtfProcessor:
    def testExtractTextFromRtf(self, tempRtfFile):
        text = extractTextFromRtf(tempRtfFile)
        assert "This is a test" in text

    def testExtractTextFromRtfFileNotFound(self):
        with pytest.raises(FileNotFoundError):
            extractTextFromRtf("non_existent_file.rtf")

    def testExtractTextFromRtfFileSizeExceeded(self, tempLargeRtfFile):
        config.maxFileSizeMB = 0.00001  # Very small limit
        try:
            with pytest.raises(FileSizeExceededError):
                extractTextFromRtf(tempLargeRtfFile)
        finally:
            config.maxFileSizeMB = 100  # Reset config

    def testExtractTextFromNonRtfFile(self, tempNonRtfFile):
        with pytest.raises(RTFEncodingError):
            extractTextFromRtf(tempNonRtfFile)

    def testExtractTextWithSpecialCharacters(self, tempRtfFileWithEncoding):
        text = extractTextFromRtf(tempRtfFileWithEncoding)
        assert "This is a test with some special characters: éàçüö" in text


class TestPipeline:
    def testProcessFile(self, tempRtfFile):
        pipeline = RtfProcessingPipeline(chunkSize=5, maxWorkers=1)
        chunks = pipeline.processFile(tempRtfFile)
        assert chunks > 0
        # Ensure that the number of vectors in the store matches the number of processed chunks
        assert pipeline.vectorStore.index.ntotal == chunks

    def testProcessFileErrorHandling(self, tempNonRtfFile):
        pipeline = RtfProcessingPipeline(chunkSize=5, maxWorkers=1)
        chunks = pipeline.processFile(tempNonRtfFile)
        assert chunks == 0
        assert pipeline.vectorStore.index.ntotal == 0

    def testProcessDirectory(self, tempRtfFile, tmp_path):
        # Create a temporary directory with multiple RTF files using tmp_path
        tempDir = tmp_path / "rtf_files"
        tempDir.mkdir()
        numFiles = 3
        for i in range(numFiles):
            file_path = tempDir / f"test_{i}.rtf"
            file_path.write_text("{\\rtf1\\ansi Test file content}")
        pipeline = RtfProcessingPipeline(chunkSize=10, maxWorkers=2)
        processedFiles = pipeline.processDirectory(tempDir)
        assert processedFiles == numFiles
        assert pipeline.vectorStore.index.ntotal == numFiles  # Each file should produce one chunk

    def testProcessDirectoryNoRtfFiles(self, tmp_path):
        tempDir = tmp_path / "empty_dir"
        tempDir.mkdir()
        pipeline = RtfProcessingPipeline(chunkSize=10, maxWorkers=2)
        processedFiles = pipeline.processDirectory(tempDir)
        assert processedFiles == 0
        assert pipeline.vectorStore.index.ntotal == 0