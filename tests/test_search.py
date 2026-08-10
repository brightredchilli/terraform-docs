"""Unit tests for query preparation and rank fusion."""

from __future__ import annotations

import pytest

from terraform_docs_mcp.search import (
    RRF_K,
    aggregate_to_documents,
    infer_provider,
    reciprocal_rank_fusion,
    to_fts_match,
)


class TestInferProvider:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("managed relational database in google cloud", "google"),
            ("allow inbound traffic on a port in gcp", "google"),
            ("how do I set up an s3 bucket in aws", "aws"),
            ("amazon rds backup retention", "aws"),
            ("gcs bucket lifecycle", "google"),
        ],
    )
    def test_detects_a_named_provider(self, query, expected):
        assert infer_provider(query) == expected

    def test_no_provider_named(self):
        assert infer_provider("delete old files in a bucket") is None

    def test_both_providers_named_is_ambiguous(self):
        """A migration question must not be silently scoped to one cloud."""
        assert infer_provider("migrating from aws to gcp") is None

    def test_substring_does_not_trigger(self):
        # Tokenised, so "awshole" or "googled" must not match.
        assert infer_provider("googled the answer") is None


class TestToFtsMatch:
    def test_plain_terms_become_or_query(self):
        assert to_fts_match("bucket lifecycle") == '"bucket" OR "lifecycle"'

    def test_empty_query_yields_empty(self):
        assert to_fts_match("") == ""
        assert to_fts_match("   ") == ""

    @pytest.mark.parametrize(
        "query",
        [
            'unbalanced " quote',
            "trailing operator AND",
            "wildcard*",
            "column:value",
            "hyphen-separated",
            "parens ( unbalanced",
            "^caret",
        ],
    )
    def test_fts5_syntax_characters_are_neutralised(self, query):
        """Raw user text must never reach FTS5 as query syntax.

        Any of these would otherwise raise sqlite3.OperationalError.
        """
        match = to_fts_match(query)
        assert '"' not in match.replace('"', "", match.count('"'))  # only our quotes
        assert match.count('"') % 2 == 0

    def test_stopwords_are_dropped(self):
        """Stopwords wreck BM25 here.

        FTS5 has no stopword list, and under OR semantics "how do I ... in a
        bucket" scores every long page containing "how"/"do"/"in" -- in
        practice the provider upgrade guides, which mention everything.
        """
        match = to_fts_match("how do I delete old files in a bucket")
        assert '"how"' not in match
        assert '"in"' not in match
        assert '"delete"' in match and '"bucket"' in match

    def test_all_stopword_query_keeps_its_terms(self):
        # Better to search literally than to return nothing at all.
        assert to_fts_match("how to do it") != ""

    def test_identifier_becomes_phrase(self):
        # The tokenizer splits on '_', so quoting makes it a phrase query and
        # matches adjacency rather than mere co-occurrence.
        assert to_fts_match("aws_instance") == '"aws_instance"'


class TestReciprocalRankFusion:
    def test_single_ranking_preserves_order(self):
        fused = reciprocal_rank_fusion([[10, 20, 30]])
        assert list(fused) == [10, 20, 30]

    def test_agreement_between_channels_wins(self):
        """A document both channels rank highly beats one only a single channel likes.

        The rank gap has to be real: with k=60 the difference between ranks 1
        and 3 is under 3%, so consensus only overtakes a lone top hit when the
        dissenting channel ranks it far down.
        """
        channel_a = [1, 2] + list(range(100, 130))
        channel_b = [3, 2] + list(range(200, 228)) + [1]  # 1 is dead last here
        fused = reciprocal_rank_fusion([channel_a, channel_b])
        assert max(fused, key=lambda k: fused[k]) == 2

    def test_top_of_one_channel_survives_absence_from_the_other(self):
        # The complement of the case above: a strong single-channel hit still
        # ranks, it just does not dominate.
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]])
        assert set(fused) == {1, 2, 3}
        assert fused[1] == pytest.approx(fused[3])

    def test_scores_use_rrf_formula(self):
        fused = reciprocal_rank_fusion([[7]])
        assert fused[7] == pytest.approx(1.0 / (RRF_K + 1))

    def test_empty_channels(self):
        assert reciprocal_rank_fusion([[], []]) == {}


class TestAggregateToDocuments:
    @staticmethod
    def _chunks(mapping):
        return {
            cid: {"doc_id": doc, "heading_path": f"h{cid}", "snippet": f"s{cid}"}
            for cid, doc in mapping.items()
        }

    def test_collapses_chunks_of_one_document(self):
        fused = {1: 0.5, 2: 0.4}
        out = aggregate_to_documents(fused, self._chunks({1: "a", 2: "a"}), limit=10)
        assert len(out) == 1
        assert out[0]["doc_id"] == "a"

    def test_best_chunk_supplies_the_snippet(self):
        fused = {1: 0.5, 2: 0.4}
        out = aggregate_to_documents(fused, self._chunks({1: "a", 2: "a"}), limit=10)
        assert out[0]["snippet"] == "s1"

    def test_multiple_hits_boost_a_document(self):
        # 'a' wins on a single strong chunk; 'b' has two weaker ones whose
        # combined weight should overtake it.
        fused = {1: 0.50, 2: 0.49, 3: 0.48}
        out = aggregate_to_documents(fused, self._chunks({1: "a", 2: "b", 3: "b"}), limit=10)
        assert [e["doc_id"] for e in out] == ["b", "a"]

    def test_respects_limit(self):
        fused = {i: 1.0 / i for i in range(1, 11)}
        chunks = self._chunks({i: f"doc{i}" for i in range(1, 11)})
        assert len(aggregate_to_documents(fused, chunks, limit=3)) == 3

    def test_long_documents_cannot_win_by_volume(self):
        """Only a capped number of runner-up chunks may contribute.

        Summing over every matching chunk rewards length, not relevance: the
        12,000-word AWS upgrade guide contributes ~113 chunks and was beating
        the correct resource page on unrelated queries.
        """
        # 'long' has ten mediocre chunks; 'short' has one strong chunk.
        fused = {0: 0.40}
        mapping = {0: "short"}
        for i in range(1, 11):
            fused[i] = 0.30
            mapping[i] = "long"
        out = aggregate_to_documents(fused, self._chunks(mapping), limit=5)
        assert out[0]["doc_id"] == "short"

    def test_ignores_unknown_chunk_ids(self):
        out = aggregate_to_documents({999: 1.0}, self._chunks({1: "a"}), limit=5)
        assert out == []
