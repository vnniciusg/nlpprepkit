import pytest
from nlpprepkit.pipeline import Pipeline
from nlpprepkit.core.exceptions import InvalidInputTypeError, ProfilingNotEnabledError, StepNotCallableError


def test_get_profile_summary_without_profiling():
    """test that get_profile_summary raises ValueError when profiling is not enabled."""
    pipeline = Pipeline(enable_profiling=False)
    with pytest.raises(ProfilingNotEnabledError, match="Profiling is not enabled. Set enable_profiling=True to enable profiling."):
        pipeline.get_profile_summary()


def test_get_profile_summary_with_profiling():
    """test that get_profile_summary returns a valid summary when profiling is enabled."""
    pipeline = Pipeline(enable_profiling=True)

    @pipeline.profiler
    def dummy_step(text: str) -> str:
        return text.upper()

    pipeline.add_step(dummy_step)

    pipeline.process("test")

    summary = pipeline.get_profile_summary()

    assert isinstance(summary, dict), "Summary should be a dictionary."
    assert "dummy_step" in summary["steps"], "Summary should include the dummy_step."
    assert "total_time" in summary["steps"]["dummy_step"], "Summary should include total_time for dummy_step."
    assert summary["steps"]["dummy_step"]["total_time"] > 0, "Total time should be greater than 0."


def dummy_step(text: str) -> str:
    return text.upper()


def test_load_pipeline():
    """Test loading a pipeline from a file."""
    pipeline = Pipeline()

    pipeline.add_step(dummy_step)

    filepath = "test_pipeline.pkl"
    pipeline.save(filepath)

    loaded_pipeline = Pipeline.load(filepath)

    assert len(loaded_pipeline.steps) == len(pipeline.steps), "Loaded pipeline should have the same number of steps."
    assert loaded_pipeline.steps[0].__name__ == pipeline.steps[0].__name__, "Loaded step should match the original step."

    import os

    os.remove(filepath)


def step1(text: str) -> str:
    return text.lower()


def step2(text: str) -> str:
    return text[::-1]


def test_pipeline_with_multiple_steps():
    """Test pipeline with multiple steps."""
    pipeline = Pipeline()

    pipeline.add_step(step1)
    pipeline.add_step(step2)

    result = pipeline.process("Test")

    assert result == "tset", "Pipeline should process text through all steps in order."


def test_save_and_load_pipeline_with_multiple_steps():
    """Test saving and loading a pipeline with multiple steps."""
    pipeline = Pipeline()

    pipeline.add_step(step1)
    pipeline.add_step(step2)

    filepath = "test_pipeline_multi.pkl"
    pipeline.save(filepath)

    loaded_pipeline = Pipeline.load(filepath)

    assert len(loaded_pipeline.steps) == len(pipeline.steps), "Loaded pipeline should have the same number of steps."
    assert loaded_pipeline.steps[0].__name__ == pipeline.steps[0].__name__, "First step should match the original step."
    assert loaded_pipeline.steps[1].__name__ == pipeline.steps[1].__name__, "Second step should match the original step."

    result = loaded_pipeline.process("Test")

    assert result == "tset", "Loaded pipeline should process text through all steps in order."

    import os

    os.remove(filepath)


def test_pipeline_with_no_steps():
    """Test pipeline with no steps."""
    pipeline = Pipeline()

    result = pipeline.process("Test")

    assert result == "Test", "Pipeline with no steps should return the input unchanged."


def test_profiling_with_multiple_steps():
    """Test profiling with multiple steps."""
    pipeline = Pipeline(enable_profiling=True)

    @pipeline.profiler
    def step1(text: str) -> str:
        return text.lower()

    @pipeline.profiler
    def step2(text: str) -> str:
        return text[::-1]

    pipeline.add_step(step1)
    pipeline.add_step(step2)

    pipeline.process("Test")

    summary = pipeline.get_profile_summary()

    assert "step1" in summary["steps"], "Summary should include step1."
    assert "step2" in summary["steps"], "Summary should include step2."
    assert summary["steps"]["step1"]["total_time"] > 0, "Total time for step1 should be greater than 0."
    assert summary["steps"]["step2"]["total_time"] > 0, "Total time for step2 should be greater than 0."


def test_clear_steps():
    """Test clearing all steps from the pipeline."""
    pipeline = Pipeline()

    def step1(text: str) -> str:
        return text.lower()

    pipeline.add_step(step1)

    assert len(pipeline.steps) == 1, "Pipeline should have one step."

    pipeline.clear_steps()

    assert len(pipeline.steps) == 0, "Pipeline should have no steps after clearing."


def test_insert_step():
    """Test inserting a step at a specific position."""
    pipeline = Pipeline()

    def step1(text: str) -> str:
        return text.lower()

    def step2(text: str) -> str:
        return text[::-1]

    pipeline.add_step(step1)
    pipeline.insert_step(0, step2)

    result = pipeline.process("Test")

    assert result == "tset", "Pipeline should process text through steps in the correct order."


def test_remove_step():
    """Test removing a step by index."""
    pipeline = Pipeline()

    def step1(text: str) -> str:
        return text.lower()

    def step2(text: str) -> str:
        return text[::-1]

    pipeline.add_step(step1)
    pipeline.add_step(step2)

    pipeline.remove_step(0)

    result = pipeline.process("Test")

    assert result == "tseT", "Pipeline should process text through the remaining steps."


def test_get_profile_summary_without_profiling():
    """Test that get_profile_summary raises an error when profiling is not enabled."""
    pipeline = Pipeline(enable_profiling=False)

    with pytest.raises(ValueError, match="Profiling is not enabled. Set enable_profiling=True to enable profiling."):
        pipeline.get_profile_summary()


def test_save_and_load_empty_pipeline():
    """Test saving and loading an empty pipeline."""
    pipeline = Pipeline()

    filepath = "empty_pipeline.pkl"
    pipeline.save(filepath)

    loaded_pipeline = Pipeline.load(filepath)

    assert len(loaded_pipeline.steps) == 0, "Loaded pipeline should have no steps."

    import os

    os.remove(filepath)


def test_pipeline_callable():
    """Test calling the pipeline instance directly."""
    pipeline = Pipeline()

    def step1(text: str) -> str:
        return text.lower()

    pipeline.add_step(step1)

    result = pipeline("Test")

    assert result == "test", "Calling the pipeline directly should process the input."


def test_add_step_with_caching():
    """Test adding a step with caching enabled."""
    pipeline = Pipeline()

    call_count = {"count": 0}

    def step_with_cache(text: str) -> str:
        call_count["count"] += 1
        return text.upper()

    pipeline.add_step(step_with_cache, use_cache=True)

    pipeline.process("test")
    pipeline.process("test")

    assert call_count["count"] == 1, "Step with caching should only be called once for the same input."


def test_process_with_list_input():
    """Test processing a list of strings."""
    pipeline = Pipeline()

    def step1(text: str) -> str:
        return text.lower()

    def step2(text: str) -> str:
        return text[::-1]

    pipeline.add_step(step1)
    pipeline.add_step(step2)

    input_data = ["Test", "Pipeline", "Processing"]
    result = pipeline.process(input_data)

    expected = ["tset", "enilepip", "gnissecorp"]
    assert result == expected, f"Expected {expected}, but got {result}"


def test_add_step_not_callable():
    """Test that adding a non-callable step raises StepNotCallableError."""
    pipeline = Pipeline()

    with pytest.raises(StepNotCallableError, match="The step '123' must be a callable function."):
        pipeline.add_step(123)


def test_process_invalid_input_type():
    """Test that processing invalid input raises InvalidInputTypeError."""
    pipeline = Pipeline()

    with pytest.raises(InvalidInputTypeError, match="Invalid input type: int. Input must be a string or a list of strings."):
        pipeline.process(123)


def test_get_profile_summary_without_profiling():
    """Test that accessing profiling summary without enabling profiling raises ProfilingNotEnabledError."""
    pipeline = Pipeline(enable_profiling=False)

    with pytest.raises(ProfilingNotEnabledError, match="Profiling is not enabled. Set enable_profiling=True to enable profiling."):
        pipeline.get_profile_summary()
