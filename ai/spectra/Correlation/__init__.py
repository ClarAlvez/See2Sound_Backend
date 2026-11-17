"""API publica Correlation v0.5.0, com aliases das versoes anteriores."""
from .config import CorrelationConfig
from .engine import CorrelationEngine
from .entities import ActiveTrack, PersonDetection, PersonEntity
from .memory import EntityMemory
from .tracker import PersonTracker
from .reconciler import EntityReconciler
from .object_entities import ActiveObjectTrack, ObjectDetection, ObjectEntity
from .object_memory import ObjectMemory
from .object_tracker import ObjectTracker
from .object_adapter import CorrelationObjectAdapter
from .object_association import ObjectAssociationMemory, PersonObjectAssociator
from .action_adapter import CorrelationActionAdapter
from .action_memory import ActionMemory
from .feature_vectors import AttributeVectorMemory
from .context_memory import ContextMemory
from .presence_memory import PresenceMemory
from .speech_identity import SpeechIdentityResolver, detect_mentions
from .pipeline_integration import correlate_spectra_with_speech
from .requesters.transcription import TranscriptionRequester, extract_speech_analysis
from .pipeline_adapter import flatten_spectra_outputs, from_pipeline_output
from .inference_bridge import SpectraInferenceBridge
from .encoders import IdentityEncoder, ResNetIdentityEncoder, create_identity_encoder
from .requesters import (PersonRequester, ObjectRequester, ActionsRequester,
                         SceneRequester, AtmosphereRequester)

__all__ = [
    'CorrelationConfig', 'CorrelationEngine', 'PersonDetection', 'PersonEntity',
    'ActiveTrack', 'EntityMemory', 'PersonTracker', 'EntityReconciler',
    'ObjectDetection', 'ObjectEntity', 'ActiveObjectTrack', 'ObjectMemory',
    'ObjectTracker', 'CorrelationObjectAdapter', 'PersonObjectAssociator',
    'ObjectAssociationMemory', 'IdentityEncoder', 'ResNetIdentityEncoder',
    'create_identity_encoder', 'CorrelationActionAdapter', 'ActionMemory',
    'AttributeVectorMemory', 'ContextMemory', 'PresenceMemory',
    'correlate_spectra_with_speech', 'SpeechIdentityResolver', 'TranscriptionRequester', 'extract_speech_analysis', 'detect_mentions',
    'SpectraInferenceBridge', 'flatten_spectra_outputs', 'from_pipeline_output', 'PersonRequester', 'ObjectRequester',
    'ActionsRequester', 'SceneRequester', 'AtmosphereRequester',
]

from ai.spectra.Correlation.audio import SpeakerDiarizer, LightASDRunner, SpeakerLinkMemory, SpeakerPipeline
