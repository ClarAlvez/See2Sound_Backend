"""Person output contract with legacy age logits; no checkpoint inference."""
import importlib.util
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def predictor(tmp_path, monkeypatch):
    labels_module = ModuleType('ai.spectra.Person.labels')
    labels_module.__dict__.update(runpy.run_path(str(ROOT / 'ai/spectra/Person/labels.py')))
    checkpoint_labels = ['person', 'child', 'adult', 'elderly', 'hat']
    class Scores(list):
        def squeeze(self, *args): return self
        def cpu(self): return self
    class Model:
        def __init__(self, output_size, **kwargs): self.output_size = output_size
        def to(self, *args): return self
        def load_state_dict(self, state): pass
        def eval(self): pass
        def __call__(self, image): return Scores([.9, .99, .98, .97, .8])
    torch = ModuleType('torch')
    torch.Tensor = object
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.load = lambda *a, **k: {'labels': checkpoint_labels, 'model_state_dict': {}}
    torch.sigmoid = lambda values: values
    torch.no_grad = nullcontext
    model_module = ModuleType('ai.spectra.Person.model')
    model_module.SpectraPersonNet = Model
    transforms = ModuleType('ai.spectra.data.transforms')
    transforms.get_test_transforms = lambda **kwargs: None
    for name, module in [('torch', torch), ('ai.spectra.Person.labels', labels_module),
                          ('ai.spectra.Person.model', model_module), ('ai.spectra.data.transforms', transforms)]:
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location('person_output_contract', ROOT / 'ai/spectra/Person/inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    value = module.PersonPredictor(str(checkpoint))
    monkeypatch.setattr(value, '_load_image_as_tensor', lambda path: None)
    image = tmp_path / 'frame.jpg'
    image.touch()
    return value, image, checkpoint_labels


def test_standard_and_grouped_outputs_exclude_age(predictor):
    value, image, _ = predictor
    result = value.predict_frame(str(image), threshold=0, top_k=2, group_by_category=True)
    assert result['predictions'] == [{'label': 'person', 'score': .9}, {'label': 'hat', 'score': .8}]
    assert result['grouped_predictions']['person'] == result['predictions']


def test_top_labels_exclude_age_before_top_k(predictor):
    value, image, _ = predictor
    result = value.predict_top_labels(str(image), top_k=2)
    assert result['top_predictions'] == [{'label': 'person', 'score': .9}, {'label': 'hat', 'score': .8}]


def test_legacy_checkpoint_indices_and_output_size_preserved(predictor):
    value, _, labels = predictor
    assert value.labels == labels
    assert value.model.output_size == len(labels)


def test_declared_labels_have_no_age():
    labels = runpy.run_path(str(ROOT / 'ai/spectra/Person/labels.py'))['LABELS']
    assert not {'child', 'adult', 'elderly'}.intersection(labels)
