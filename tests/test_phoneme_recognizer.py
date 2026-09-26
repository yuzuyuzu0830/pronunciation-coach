from unittest.mock import MagicMock, patch

from pronunciation_coach.phoneme_recognizer import _load_processor


def test_load_processor_builds_tokenizer_and_feature_extractor_explicitly() -> None:
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_feature_extractor = MagicMock(name="feature_extractor")
    fake_processor = MagicMock(name="processor")

    with (
        patch(
            "pronunciation_coach.phoneme_recognizer.Wav2Vec2PhonemeCTCTokenizer"
        ) as tokenizer_cls,
        patch(
            "pronunciation_coach.phoneme_recognizer.Wav2Vec2FeatureExtractor"
        ) as feature_extractor_cls,
        patch(
            "pronunciation_coach.phoneme_recognizer.Wav2Vec2Processor"
        ) as processor_cls,
    ):
        tokenizer_cls.from_pretrained.return_value = fake_tokenizer
        feature_extractor_cls.from_pretrained.return_value = fake_feature_extractor
        processor_cls.return_value = fake_processor

        result = _load_processor("some/model-name")

        tokenizer_cls.from_pretrained.assert_called_once_with("some/model-name")
        feature_extractor_cls.from_pretrained.assert_called_once_with("some/model-name")
        processor_cls.assert_called_once_with(
            feature_extractor=fake_feature_extractor, tokenizer=fake_tokenizer
        )
        assert result is fake_processor
