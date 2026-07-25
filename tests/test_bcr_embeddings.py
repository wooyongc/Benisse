"""Contracts for the optional ESM-2 CDR3H embedding backend."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bcr_embeddings as embeddings
import benisse_pipeline as pipeline
import benisse_preprocessing as preprocessing


def _write_input(path: Path, sequences) -> Path:
    pd.DataFrame(
        {
            "contigs": [f"contig_{index}" for index in range(len(sequences))],
            "cdr3": sequences,
        }
    ).to_csv(path, index=False)
    return path


def test_achilles_dispatch_delegates_without_changing_arguments(tmp_path, monkeypatch):
    calls = []
    expected = pd.DataFrame({0: [1.0], "index": ["CAR"]})

    def fake_encode(input_data, output_data, cuda):
        calls.append((input_data, output_data, cuda))
        return expected

    monkeypatch.setitem(
        __import__("sys").modules,
        "AchillesEncoder",
        SimpleNamespace(encode_bcr=fake_encode),
    )
    output = tmp_path / "encoded.csv"
    result = embeddings.encode_bcr_embeddings("input.csv", output, cuda=False)

    assert result is expected
    assert calls == [("input.csv", output, False)]
    assert not embeddings.esm2_metadata_path(output).exists()


def test_backend_specific_arguments_are_rejected_for_achilles(tmp_path):
    with pytest.raises(ValueError, match="model is only valid"):
        embeddings.encode_bcr_embeddings(
            "input.csv", tmp_path / "out.csv", model=embeddings.DEFAULT_ESM2_MODEL
        )
    with pytest.raises(ValueError, match="offline is only valid"):
        embeddings.encode_bcr_embeddings(
            "input.csv", tmp_path / "out.csv", offline=True
        )


def test_esm2_input_is_amino_acid_sorted_deduplicated_and_validated(tmp_path):
    first = tmp_path / "b.csv"
    second = tmp_path / "a.csv"
    _write_input(first, [" cAr ", "CASS", "CAR"])
    _write_input(second, ["CAW"])

    assert embeddings._read_esm2_sequences(tmp_path) == ["CAR", "CASS", "CAW"]

    _write_input(tmp_path / "bad.csv", ["CA*R"])
    with pytest.raises(ValueError, match="noncanonical.*bad.csv"):
        embeddings._read_esm2_sequences(tmp_path / "bad.csv")

    _write_input(tmp_path / "long.csv", ["A" * 1023])
    with pytest.raises(ValueError, match="at most 1022"):
        embeddings._read_esm2_sequences(tmp_path / "long.csv")

    _write_input(tmp_path / "empty.csv", ["   "])
    with pytest.raises(ValueError, match="empty CDR3"):
        embeddings._read_esm2_sequences(tmp_path / "empty.csv")

    missing = pd.DataFrame(
        {"contigs": ["contig_0", "contig_1"], "cdr3": ["CAR", None]}
    )
    missing.to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match=r"missing required value.*cdr3.*missing.csv:3"):
        embeddings._read_esm2_sequences(tmp_path / "missing.csv")


def test_esm2_missing_optional_dependency_is_actionable(monkeypatch):
    def missing(_name):
        raise embeddings.metadata.PackageNotFoundError

    monkeypatch.setattr(embeddings.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="fair-esm==2.0.0"):
        embeddings._optional_esm_modules()


class _FakeAlphabet:
    padding_idx = 0
    _tokens = {letter: index + 1 for index, letter in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def get_batch_converter(self):
        def convert(items):
            import torch

            width = max(len(sequence) for _, sequence in items) + 2
            tokens = torch.zeros((len(items), width), dtype=torch.long)
            for row, (_, sequence) in enumerate(items):
                tokens[row, 0] = 100
                for column, residue in enumerate(sequence, start=1):
                    tokens[row, column] = self._tokens[residue]
                tokens[row, len(sequence) + 1] = 200
            return [item[0] for item in items], [item[1] for item in items], tokens

        return convert


class _FakeModel:
    num_layers = 12
    embed_dim = 480

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, tokens, repr_layers, return_contacts):
        import torch

        assert repr_layers == [12]
        assert return_contacts is False
        batch, width = tokens.shape
        result = torch.zeros((batch, width, self.embed_dim), dtype=torch.float32)
        result[:, :, 0] = tokens
        result[:, :, 1] = torch.arange(width, dtype=torch.float32)
        return {"representations": {12: result}}


def _install_fake_esm(monkeypatch, tmp_path):
    import torch

    cache_root = tmp_path / "torch-hub"
    checkpoint_dir = cache_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for suffix in (".pt", "-contact-regression.pt"):
        (checkpoint_dir / f"{embeddings.DEFAULT_ESM2_MODEL}{suffix}").write_bytes(
            f"checkpoint{suffix}".encode()
        )
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache_root))
    fake_esm = SimpleNamespace(
        pretrained=SimpleNamespace(
            esm2_t12_35M_UR50D=lambda: (_FakeModel(), _FakeAlphabet())
        )
    )
    monkeypatch.setattr(
        embeddings,
        "_optional_esm_modules",
        lambda: (fake_esm, torch, "2.0.0"),
    )
    return cache_root


def test_esm2_pooling_normalization_schema_and_provenance(tmp_path, monkeypatch):
    cache_root = _install_fake_esm(monkeypatch, tmp_path)
    input_csv = _write_input(tmp_path / "input.csv", ["AC", "A", "AC"])
    output_csv = tmp_path / "encoded.csv"

    result = embeddings.encode_bcr_embeddings(
        input_csv,
        output_csv,
        backend="esm2",
        model=embeddings.DEFAULT_ESM2_MODEL,
        batch_size=2,
        offline=True,
    )

    assert list(result.columns) == list(range(480)) + ["index"]
    assert result["index"].tolist() == ["A", "AC"]
    assert result[list(range(480))].to_numpy().dtype == np.float64
    np.testing.assert_allclose(
        np.linalg.norm(result[list(range(480))].to_numpy(), axis=1),
        1,
        rtol=0,
        atol=2e-16,
    )
    # Fake residue representations are [token_id, token_position, 0...].
    expected_a = np.array([1.0, 1.0]) / np.sqrt(2)
    expected_ac = np.array([1.5, 1.5]) / np.sqrt(4.5)
    np.testing.assert_allclose(
        result.iloc[0, :2].to_numpy(dtype=float), expected_a, rtol=0, atol=1e-7
    )
    np.testing.assert_allclose(
        result.iloc[1, :2].to_numpy(dtype=float), expected_ac, rtol=0, atol=1e-7
    )

    written = pd.read_csv(output_csv)
    assert written.columns.tolist() == [
        "Unnamed: 0", *map(str, range(480)), "index"
    ]
    sidecar = json.loads(embeddings.esm2_metadata_path(output_csv).read_text())
    assert sidecar["model"] == embeddings.DEFAULT_ESM2_MODEL
    assert sidecar["embedding_dimension"] == 480
    assert sidecar["pooling"] == "mean_residue"
    assert sidecar["normalization"] == "l2"
    assert sidecar["checkpoints"]["model"]["path"].startswith("checkpoints/")
    assert len(sidecar["checkpoints"]["model"]["sha256"]) == 64
    assert Path(cache_root / sidecar["checkpoints"]["model"]["path"]).is_file()
    assert pipeline._encoder_model_identity("esm2", output_csv) == sidecar


def test_esm2_offline_requires_both_cache_files(tmp_path, monkeypatch):
    import torch

    cache_root = tmp_path / "torch-hub"
    (cache_root / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache_root))
    fake_esm = SimpleNamespace(pretrained=SimpleNamespace())
    monkeypatch.setattr(
        embeddings,
        "_optional_esm_modules",
        lambda: (fake_esm, torch, "2.0.0"),
    )
    input_csv = _write_input(tmp_path / "input.csv", ["CAR"])

    with pytest.raises(RuntimeError, match="offline.*cached checkpoint"):
        embeddings.encode_bcr_embeddings(
            input_csv,
            tmp_path / "encoded.csv",
            backend="esm2",
            offline=True,
        )


def test_zero_norm_embedding_is_rejected(tmp_path, monkeypatch):
    _install_fake_esm(monkeypatch, tmp_path)

    class ZeroModel(_FakeModel):
        def __call__(self, tokens, repr_layers, return_contacts):
            import torch

            return {
                "representations": {
                    12: torch.zeros(
                        (tokens.shape[0], tokens.shape[1], self.embed_dim),
                        dtype=torch.float32,
                    )
                }
            }

    fake_esm, torch, version = embeddings._optional_esm_modules()
    fake_esm.pretrained.esm2_t12_35M_UR50D = lambda: (ZeroModel(), _FakeAlphabet())
    monkeypatch.setattr(
        embeddings, "_optional_esm_modules", lambda: (fake_esm, torch, version)
    )
    input_csv = _write_input(tmp_path / "input.csv", ["CAR"])
    with pytest.raises(RuntimeError, match="zero-norm"):
        embeddings.encode_bcr_embeddings(
            input_csv,
            tmp_path / "encoded.csv",
            backend="esm2",
            offline=True,
        )


def test_dynamic_feature_columns_and_preprocessing(tmp_path):
    encoded = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "0": [0.0, 1.0],
            "1": [1.0, 0.0],
            "2": [0.5, 0.25],
            "index": ["CA", "CB"],
        }
    )
    assert preprocessing.embedding_feature_columns(encoded) == ("0", "1", "2")
    assert preprocessing._encoder_lookup(encoded).shape == (2, 3)

    with pytest.raises(ValueError, match="contiguous"):
        preprocessing.embedding_feature_columns(
            encoded.drop(columns="1").rename(columns={"2": "3"})
        )
    with pytest.raises(ValueError, match="non-feature"):
        preprocessing.embedding_feature_columns(encoded.assign(notes="bad"))

    expression = pd.DataFrame(
        np.random.default_rng(13).normal(size=(12, 4)),
        index=[f"g{i}" for i in range(12)],
        columns=["a", "b", "c", "d"],
    )
    annotation = pd.DataFrame(
        {
            "v_gene": ["V1", "V1"],
            "j_gene": ["J1", "J1"],
            "cdr3": ["CA", "CB"],
        },
        index=["V1_CA_J1", "V1_CB_J1"],
    )
    prepared = preprocessing.initialize_core_inputs(
        expression,
        ["V1_CA_J1", "V1_CA_J1", "V1_CB_J1", "V1_CB_J1"],
        annotation,
        encoded[["0", "1", "2"]].to_numpy(),
        pipeline.BenisseParams(lambda2=4, m=2).hyperparameters(),
    )
    assert prepared.embedding.shape == (2, 3)
    assert prepared.phi.shape == (2, 2)


def test_pipeline_encoder_keyword_propagation(tmp_path, monkeypatch):
    calls = []

    def fake_encode(input_data, output_data, **kwargs):
        calls.append((input_data, output_data, kwargs))
        Path(output_data).write_text(",0,index\n0,1.0,CAR\n")

    monkeypatch.setattr(embeddings, "encode_bcr_embeddings", fake_encode)
    output = pipeline.encode_bcr_csv(
        "input.csv",
        tmp_path / "encoded.csv",
        cuda=False,
        embedding_backend="esm2",
        embedding_model="esm2_t6_8M_UR50D",
        embedding_batch_size=7,
        embedding_offline=True,
    )
    assert output == tmp_path / "encoded.csv"
    assert calls[0][2] == {
        "backend": "esm2",
        "model": "esm2_t6_8M_UR50D",
        "cuda": False,
        "batch_size": 7,
        "offline": True,
    }


def test_cli_contract():
    parsed = embeddings.parse_option(
        [
            "--input_data", "input.csv",
            "--output_data", "output.csv",
            "--backend", "esm2",
            "--model", "esm2_t6_8M_UR50D",
            "--batch_size", "8",
            "--cuda", "False",
            "--offline",
        ]
    )
    assert parsed.backend == "esm2"
    assert parsed.model == "esm2_t6_8M_UR50D"
    assert parsed.batch_size == 8
    assert parsed.cuda is False
    assert parsed.offline is True


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("BENISSE_RUN_ESM2_TESTS") != "1",
    reason="set BENISSE_RUN_ESM2_TESTS=1 for the real ESM-2 integration",
)
def test_real_esm2_is_repeatable_and_runs_corrected_pipeline(tmp_path):
    encoder_input = _write_input(
        tmp_path / "encoder_input.csv",
        ["CARAAW", "CARDQW", "CASSLW", "CATGGW"],
    )
    first = embeddings.encode_bcr_embeddings(
        encoder_input,
        tmp_path / "first.csv",
        backend="esm2",
        model=embeddings.DEFAULT_ESM2_MODEL,
        batch_size=2,
    )
    reversed_input = pd.read_csv(encoder_input).iloc[::-1]
    reversed_input.to_csv(tmp_path / "reversed.csv", index=False)
    second = embeddings.encode_bcr_embeddings(
        tmp_path / "reversed.csv",
        tmp_path / "second.csv",
        backend="esm2",
        model=embeddings.DEFAULT_ESM2_MODEL,
        batch_size=3,
    )
    assert first.shape == second.shape == (4, 481)
    np.testing.assert_allclose(
        first.iloc[:, :480],
        second.iloc[:, :480],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        np.linalg.norm(first.iloc[:, :480], axis=1),
        1,
        rtol=0,
        atol=2e-15,
    )

    cells = [f"cell-{index}" for index in range(6)]
    expression = pd.DataFrame(
        np.random.default_rng(11).normal(size=(12, 6)),
        index=[f"gene_{index}" for index in range(12)],
        columns=cells,
    )
    cdr3s = ["CARAAW", "CARDQW", "CARAAW", "CASSLW", "CATGGW", "CATGGW"]
    contigs = pd.DataFrame(
        {
            "barcode": cells,
            "is_cell": True,
            "high_confidence": True,
            "full_length": True,
            "productive": True,
            "chain": "IGH",
            "v_gene": ["V1"] * 3 + ["V2"] * 3,
            "j_gene": ["J1"] * 3 + ["J2"] * 3,
            "cdr3": cdr3s,
            "cdr3_nt": ["TGTGCC"] * 6,
            "umis": [6, 5, 4, 3, 2, 1],
        }
    )
    expression.to_csv(tmp_path / "expression.csv")
    contigs.to_csv(tmp_path / "contigs.csv", index=False)
    result = pipeline.run_encoded_csv_pipeline(
        tmp_path / "expression.csv",
        tmp_path / "contigs.csv",
        tmp_path / "first.csv",
        tmp_path / "result",
        params=pipeline.BenisseParams(lambda2=6, m=2),
        generate_plots=False,
        max_nodes=10,
        provenance={
            "encoder_model": pipeline._encoder_model_identity(
                "esm2", tmp_path / "first.csv"
            )
        },
    )
    assert result.core.converged
    assert result.prepared.embedding.shape == (4, 480)
    assert np.isfinite(result.core.a_matrix).all()
