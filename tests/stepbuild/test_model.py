"""ScriptedModel is the stand-in for a real model in every harness test, so it must
be exact: replies in order, a loud error when a test asks for more than it
scripted (a runner that loops too often must fail, not hang or reuse a reply),
and a faithful record of what it was shown."""
import pytest

from stepbuild.harness.model import ScriptedModel, StepModel


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_returns_replies_in_order():
    model = ScriptedModel(["one", "two", "three"])
    assert [model.complete(_msgs(str(i))) for i in range(3)] == ["one", "two", "three"]


def test_over_call_raises_runtime_error():
    model = ScriptedModel(["only"])
    model.complete(_msgs("a"))
    with pytest.raises(RuntimeError, match="1"):
        model.complete(_msgs("b"))


def test_empty_script_raises_on_first_call():
    with pytest.raises(RuntimeError):
        ScriptedModel([]).complete(_msgs("a"))


def test_records_every_call_as_a_copy():
    model = ScriptedModel(["r1", "r2"])
    first = _msgs("first")
    model.complete(first)
    first[0]["content"] = "changed later"
    first.append({"role": "user", "content": "extra"})
    model.complete(_msgs("second"))
    assert model.calls == [_msgs("first"), _msgs("second")]


def test_name_defaults_and_can_be_set():
    assert ScriptedModel(["x"]).name == "scripted"
    assert ScriptedModel(["x"], name="ref-todo").name == "ref-todo"


def test_context_tokens_defaults_to_none_and_can_be_set():
    assert ScriptedModel(["x"]).context_tokens is None
    assert ScriptedModel(["x"], context_tokens=4096).context_tokens == 4096


def test_satisfies_the_protocol():
    model: StepModel = ScriptedModel(["x"])
    assert isinstance(model, StepModel)
    assert isinstance(model.name, str)
    assert callable(model.complete)


def test_replies_must_be_strings():
    with pytest.raises(TypeError):
        ScriptedModel([1])
    with pytest.raises(TypeError):
        ScriptedModel("a single string is not a script")
