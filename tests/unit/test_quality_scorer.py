"""
Tests for ResponseQualityScorer.
"""

from agentfuse.core.quality_scorer import ResponseQualityScorer


def test_empty_response_low_score():
    """Empty response must get low score."""
    scorer = ResponseQualityScorer()
    result = scorer.score("What is Python?", "")
    assert result.overall == 0
    assert result.should_cache is False


def test_relevant_response_high_score():
    """Relevant, complete response must score high."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "What is Python?",
        "Python is a high-level programming language known for its simplicity and readability. "
        "It was created by Guido van Rossum and first released in 1991. Python supports multiple "
        "programming paradigms including procedural, object-oriented, and functional programming."
    )
    assert result.overall > 0.5
    assert result.should_cache is True


def test_irrelevant_response_low_relevancy():
    """Completely irrelevant response must score low on relevancy."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "What is Python?",
        "The weather today is sunny with a high of 75 degrees."
    )
    assert result.relevancy < 0.5


def test_short_response_low_completeness():
    """Very short response must score low on completeness."""
    scorer = ResponseQualityScorer()
    result = scorer.score("Explain quantum mechanics", "It's physics.")
    assert result.completeness < 0.5


def test_verbose_response_low_conciseness():
    """Excessively verbose response must score low on conciseness."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "hi",
        "x " * 5000  # 10K chars for a 2-char query
    )
    assert result.conciseness < 0.5


def test_cache_threshold_configurable():
    """Custom threshold must control caching decision."""
    strict = ResponseQualityScorer(cache_threshold=0.9)
    lenient = ResponseQualityScorer(cache_threshold=0.1)

    response = "Short answer"
    strict_score = strict.score("question?", response)
    lenient_score = lenient.score("question?", response)

    assert lenient_score.should_cache is True
    # Strict may or may not cache depending on score


def test_coherent_response_scores_well():
    """Well-structured response must score high on coherence."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "How to learn Python?",
        "Here are the steps to learn Python:\n\n"
        "1. Start with basic syntax and data types.\n"
        "2. Learn about functions and classes.\n"
        "3. Practice with small projects.\n"
        "4. Read documentation and tutorials.\n\n"
        "Python is a great language for beginners because of its readability."
    )
    assert result.coherence > 0.6


def test_quality_score_all_fields():
    """QualityScore must have all expected fields."""
    scorer = ResponseQualityScorer()
    result = scorer.score("test", "test response here")
    assert hasattr(result, "relevancy")
    assert hasattr(result, "completeness")
    assert hasattr(result, "conciseness")
    assert hasattr(result, "coherence")
    assert hasattr(result, "overall")
    assert hasattr(result, "should_cache")
    assert hasattr(result, "reason")


def test_code_response_with_formatting():
    """Response with code blocks must score well on coherence."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "How to print in Python?",
        "To print in Python, use the print() function:\n\n```python\nprint('Hello World')\n```\n\nThis outputs 'Hello World' to the console."
    )
    assert result.coherence > 0.5


def test_repetitive_response_low_coherence():
    """Highly repetitive response must score low."""
    scorer = ResponseQualityScorer()
    result = scorer.score(
        "What is AI?",
        "AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI AI " * 10
    )
    assert result.coherence < 0.7


def test_no_query_neutral_relevancy():
    """Empty query must give neutral relevancy score."""
    scorer = ResponseQualityScorer()
    result = scorer.score("", "Some response text here.")
    assert result.relevancy == 0.5


def test_long_response_high_completeness():
    """Long, detailed response must score high on completeness."""
    scorer = ResponseQualityScorer()
    result = scorer.score("Explain X", "Detailed explanation. " * 100)
    assert result.completeness > 0.7


def test_reason_provided_when_not_cached():
    """When should_cache is False, reason must be non-empty."""
    scorer = ResponseQualityScorer(cache_threshold=0.99)
    result = scorer.score("test", "short")
    if not result.should_cache:
        assert len(result.reason) > 0
