import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


CNN_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = CNN_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cnn_surrogate.config import DistillationTrainingConfig
from cnn_surrogate.losses import weighted_mse_loss, distillation_loss
from cnn_surrogate.models import TeacherSurrogate, StudentSurrogate
from cnn_surrogate import training
from cnn_surrogate.training import train_teacher_model, train_student_model


def tearDownModule():
    try:
        sys.path.remove(str(SRC_DIR))
    except ValueError:
        pass


def _config(epochs=1, batch_size=2, show_progress=False, early_stopping_patience=None, device="cpu"):
    return DistillationTrainingConfig(
        data_csv="input.csv",
        output_dir="output",
        figure_dir="figures",
        temp_dir="temp",
        train_test_split=1,
        split_shuffle=False,
        random_seed=123,
        pixel_size=2.0,
        image_height=80,
        image_width=40,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        dropout=0.2,
        loss_weight_stiffness=1.5,
        loss_weight_strain=0.5,
        early_stopping_patience=early_stopping_patience,
        device=device,
        show_progress=show_progress,
        progress_description="Training distillation test",
        save_model=False,
        teacher_epochs=epochs,
        student_epochs=epochs,
        teacher_learning_rate=1.0e-3,
        student_learning_rate=1.0e-3,
        distill_weight=0.3,
    )


def _distillation_dataset(count=4):
    images = torch.zeros((count, 1, 80, 40), dtype=torch.float32)
    local_features = torch.zeros((count, 24), dtype=torch.float32)
    targets = torch.zeros((count, 2), dtype=torch.float32)
    return TensorDataset(images, local_features, targets)


class SpyTeacherSurrogate(TeacherSurrogate):
    def __init__(self, dropout=0.2):
        super(SpyTeacherSurrogate, self).__init__(dropout=dropout)
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super(SpyTeacherSurrogate, self).eval()


class DistillationLossTests(unittest.TestCase):
    def test_distillation_loss_combines_supervised_and_teacher_weighted_mse(self):
        config = _config()
        student_prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        true_target = torch.tensor([[1.5, 1.0], [2.5, 5.0]], dtype=torch.float32)
        teacher_prediction = torch.tensor([[0.0, 3.0], [4.0, 3.0]], dtype=torch.float32)

        supervised = weighted_mse_loss(
            student_prediction,
            true_target,
            config.loss_weight_stiffness,
            config.loss_weight_strain,
        )
        distilled = weighted_mse_loss(
            student_prediction,
            teacher_prediction,
            config.loss_weight_stiffness,
            config.loss_weight_strain,
        )

        actual = distillation_loss(student_prediction, true_target, teacher_prediction, config)

        self.assertTrue(torch.isclose(actual, supervised + config.distill_weight * distilled))


class DistillationTrainingTests(unittest.TestCase):
    def test_train_teacher_model_returns_teacher_and_history(self):
        model, history = train_teacher_model(_distillation_dataset(), None, _config())

        self.assertIsInstance(model, TeacherSurrogate)
        self.assertGreater(len(history), 0)
        self.assertIn("train_loss", history[0])

    def test_train_student_model_returns_student_and_history(self):
        teacher_model = TeacherSurrogate(dropout=0.2)

        model, history = train_student_model(_distillation_dataset(), None, teacher_model, _config())

        self.assertIsInstance(model, StudentSurrogate)
        self.assertGreater(len(history), 0)
        self.assertIn("train_loss", history[0])

    def test_train_student_model_evals_teacher_without_teacher_gradients(self):
        teacher_model = SpyTeacherSurrogate(dropout=0.2)

        train_student_model(_distillation_dataset(), None, teacher_model, _config())

        self.assertTrue(teacher_model.eval_called)
        for parameter in teacher_model.parameters():
            self.assertIsNone(parameter.grad)

    def test_train_teacher_model_stops_after_patience_when_validation_loss_does_not_improve(self):
        calls = []
        original_run_teacher_epoch = training._run_teacher_epoch

        def fake_run_teacher_epoch(model, loader, optimizer=None, stiffness_weight=1.0, strain_weight=1.0, device=None):
            calls.append(optimizer is not None)
            if optimizer is not None:
                return 0.5
            return 1.0

        training._run_teacher_epoch = fake_run_teacher_epoch
        try:
            _, history = train_teacher_model(
                _distillation_dataset(),
                _distillation_dataset(),
                _config(epochs=5, early_stopping_patience=1),
            )
        finally:
            training._run_teacher_epoch = original_run_teacher_epoch

        self.assertEqual([record["epoch"] for record in history], [1, 2])
        self.assertEqual(calls, [True, False, True, False])

    def test_train_student_model_stops_after_patience_when_validation_loss_does_not_improve(self):
        calls = []
        original_run_student_epoch = training._run_student_epoch

        def fake_run_student_epoch(model, teacher_model, loader, config, optimizer=None, device=None):
            calls.append(optimizer is not None)
            if optimizer is not None:
                return 0.5
            return 1.0

        teacher_model = TeacherSurrogate(dropout=0.2)
        training._run_student_epoch = fake_run_student_epoch
        try:
            _, history = train_student_model(
                _distillation_dataset(),
                _distillation_dataset(),
                teacher_model,
                _config(epochs=5, early_stopping_patience=1),
            )
        finally:
            training._run_student_epoch = original_run_student_epoch

        self.assertEqual([record["epoch"] for record in history], [1, 2])
        self.assertEqual(calls, [True, False, True, False])

    def test_train_teacher_model_passes_resolved_device_to_epoch_runner(self):
        calls = []
        original_run_teacher_epoch = training._run_teacher_epoch

        def fake_run_teacher_epoch(model, loader, optimizer=None, stiffness_weight=1.0, strain_weight=1.0, device=None):
            calls.append(torch.device(device).type)
            return 0.5

        training._run_teacher_epoch = fake_run_teacher_epoch
        try:
            train_teacher_model(_distillation_dataset(), None, _config(device="cpu"))
        finally:
            training._run_teacher_epoch = original_run_teacher_epoch

        self.assertEqual(calls, ["cpu", "cpu"])

    def test_train_student_model_passes_resolved_device_to_epoch_runner(self):
        calls = []
        original_run_student_epoch = training._run_student_epoch

        def fake_run_student_epoch(model, teacher_model, loader, config, optimizer=None, device=None):
            calls.append(torch.device(device).type)
            return 0.5

        training._run_student_epoch = fake_run_student_epoch
        try:
            train_student_model(
                _distillation_dataset(),
                None,
                TeacherSurrogate(dropout=0.2),
                _config(device="cpu"),
            )
        finally:
            training._run_student_epoch = original_run_student_epoch

        self.assertEqual(calls, ["cpu", "cpu"])


if __name__ == "__main__":
    unittest.main()
