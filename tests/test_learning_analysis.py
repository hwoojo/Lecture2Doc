from lecture2doc.learning_analysis import analyze_learning


def test_links_board_concepts_to_speech_and_keeps_each_appearance_time():
    manifest = {
        "board_pages": [{"id": "p00001", "start": 10.0, "end": 20.0}],
        "transcript": {
            "status": "complete",
            "segments": [
                {"id": "t000001", "start": 12.0, "end": 15.0, "text": "선형 변환의 정의를 알아두세요."},
                {"id": "t000002", "start": 40.0, "end": 43.0, "text": "선형변환은 시험에 자주 나옵니다."},
                {"id": "t000003", "start": 50.0, "end": 54.0, "text": "다음은 행렬 곱셈입니다."},
            ],
        },
        "ocr": {
            "pages": [{
                "board_page_id": "p00001",
                "regions": [{"text": "선형 변환", "score": 0.98}],
            }],
        },
    }

    result = analyze_learning(manifest)
    page = result["pages"][0]
    concept = next(item for item in result["concepts"] if item["name"] == "선형 변환")

    assert result["status"] == "complete"
    assert page["importance_score"] is not None
    assert page["importance_level"] == "high"
    assert "transcript_links" not in page
    assert {item["segment_id"] for item in concept["speech_occurrences"]} == {"t000001", "t000002"}
    assert concept["board_appearances"][0]["start"] == 10.0


def test_transcript_only_analysis_is_partial_and_uses_page_time():
    manifest = {
        "board_pages": [{"id": "p00002", "start": 5.0, "end": 18.0}],
        "transcript": {
            "segments": [
                {"id": "t000010", "start": 6.0, "end": 9.0, "text": "고유값의 정의를 설명하겠습니다."},
                {"id": "t000011", "start": 12.0, "end": 15.0, "text": "고유값은 중요하니 기억하세요."},
            ],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["importance_score"] is not None
    assert any(item["name"] == "고유값" for item in result["concepts"])


def test_ocr_without_speech_does_not_invent_importance_score():
    manifest = {
        "board_pages": [{"id": "p00003", "start": 2.0, "end": 12.0}],
        "transcript": {"segments": []},
        "ocr": {
            "pages": [{
                "board_page_id": "p00003",
                "regions": [{"text": "미분 방정식", "score": 0.91}],
            }],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["importance_score"] is None
    assert result["pages"][0]["importance_level"] == "unrated"
    assert result["concepts"][0]["board_appearances"][0]["start"] == 2.0


def test_no_board_or_transcript_returns_insufficient_evidence():
    result = analyze_learning({})

    assert result["status"] == "insufficient_evidence"
    assert result["pages"] == []
    assert result["concepts"] == []


def test_low_confidence_ocr_text_is_not_treated_as_board_evidence():
    manifest = {
        "board_pages": [{"id": "p00004", "start": 2.0, "end": 12.0}],
        "transcript": {
            "segments": [
                {"id": "t000020", "start": 3.0, "end": 6.0, "text": "고유값은 중요하니 기억하세요."},
            ],
        },
        "ocr": {
            "pages": [{
                "board_page_id": "p00004",
                "regions": [{"text": "고유값", "score": 0.45, "review": True}],
            }],
        },
    }

    result = analyze_learning(manifest)

    assert result["status"] == "partial"
    assert result["pages"][0]["ocr_available"] is False
    assert result["pages"][0]["importance_score"] is not None
    concept = next(item for item in result["concepts"] if item["name"] == "고유값")
    assert concept["board_appearances"] == []



def test_discourse_fillers_are_not_extracted_as_concepts():
    from lecture2doc.learning_analysis import _extract_terms

    terms = _extract_terms("한번 더 설명하면 이것도 중요한 내용입니다.")
    assert "한번" not in terms
    assert "것도" not in terms
    assert "설명" not in terms
    assert "설명하면" not in terms
    assert "내용입니다" not in terms
    assert "중요한" in terms


def test_ai_selection_keeps_only_supported_topic_concepts():
    from lecture2doc.learning_analysis import _apply_ai_selection

    result = {
        "pages": [{
            "concepts": [
                {"name": "미분방정식", "importance_score": 80},
                {"name": "한번", "importance_score": 20},
            ],
        }],
        "concepts": [
            {
                "id": "c1", "name": "미분방정식", "importance_score": 80,
                "speech_occurrences": [
                    {"segment_id": "t1", "start": 1, "end": 2, "text": "미분방정식입니다."}
                ],
                "board_appearances": [{"board_page_id": "p1", "start": 0, "end": 5}],
                "board_page_ids": ["p1"], "reasons": [],
            },
            {
                "id": "c2", "name": "한번", "importance_score": 20,
                "speech_occurrences": [], "board_appearances": [],
                "board_page_ids": [], "reasons": [],
            },
        ],
    }
    curated = _apply_ai_selection(
        result,
        {
            "topic": "미분방정식",
            "concepts": [{
                "name": "미분방정식",
                "source_terms": ["미분방정식"],
                "reason": "강의의 핵심 이론으로 반복 설명됩니다.",
                "relevance": 96,
            }],
        },
        "qwen3-vl:4b",
    )
    assert [item["name"] for item in curated["concepts"]] == ["미분방정식"]
    assert curated["concepts"][0]["speech_occurrences"][0]["segment_id"] == "t1"
    assert curated["ai_curation"]["status"] == "complete"
    assert curated["ai_curation"]["topic"] == "미분방정식"
    assert len(curated["pages"][0]["concepts"]) == 1


def test_fallback_concepts_filters_filler_words_and_limits_count():
    from lecture2doc.learning_analysis import _fallback_concepts

    concepts = [
        {"name": "한번", "board_page_ids": ["p1"], "speech_occurrences": [{"segment_id": "t1"}]},
        {"name": "것도", "board_page_ids": ["p1"], "speech_occurrences": [{"segment_id": "t2"}]},
    ] + [
        {"name": f"개념{i}", "board_page_ids": ["p1"], "speech_occurrences": []}
        for i in range(10)
    ]
    selected = _fallback_concepts(concepts)
    assert len(selected) <= 5
    assert all(item["name"] not in {"한번", "것도"} for item in selected)



def test_curation_uses_local_qwen_without_openai_key(monkeypatch):
    from lecture2doc import learning_analysis, qwen_vision

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LECTURE2DOC_QWEN_MODEL", "qwen3-vl:4b")
    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": True, "model": model},
    )
    monkeypatch.setattr(
        learning_analysis, "_ai_request",
        lambda payload, model: {
            "topic": "미분방정식",
            "concepts": [{
                "name": "미분방정식",
                "source_terms": ["미분방정식"],
                "reason": "강의 전체에서 반복적으로 정의와 풀이를 다룹니다.",
                "relevance": 95,
            }],
        },
    )
    result = {
        "pages": [{
            "concepts": [{
                "name": "미분방정식", "importance_score": 80,
            }],
        }],
        "concepts": [{
            "id": "c1",
            "name": "미분방정식",
            "importance_score": 80,
            "speech_occurrences": [
                {"segment_id": "t1", "start": 1, "end": 3, "text": "미분방정식의 정의입니다."}
            ],
            "board_appearances": [{"board_page_id": "p1", "start": 0, "end": 4}],
            "board_page_ids": ["p1"],
            "reasons": ["음성에서 1회 언급"],
        }],
    }
    manifest = {
        "transcript": {"segments": [{"id": "t1", "start": 1, "end": 3, "text": "미분방정식의 정의입니다."}]},
        "ocr": {"pages": [{"board_page_id": "p1", "text": "미분방정식", "regions": [{"text": "미분방정식", "score": 0.95}]}]},
    }

    curated = learning_analysis._curate_concepts_with_ai(manifest, result)

    assert curated["ai_curation"]["status"] == "complete"
    assert curated["ai_curation"]["provider"] == "ollama-qwen"
    assert curated["ai_curation"]["model"] == "qwen3-vl:4b"
    assert curated["ai_curation"]["topic"] == "미분방정식"
    assert [item["name"] for item in curated["concepts"]] == ["미분방정식"]


def test_curation_falls_back_when_local_qwen_is_not_installed(monkeypatch):
    from lecture2doc import learning_analysis, qwen_vision

    monkeypatch.setattr(
        qwen_vision, "ollama_status",
        lambda model=None: {"installed": False, "model": model},
    )
    result = {
        "pages": [{"concepts": [{"name": "고유값"}]}],
        "concepts": [{
            "id": "c1", "name": "고유값", "importance_score": 80,
            "speech_occurrences": [
                {"segment_id": "t1", "start": 1, "end": 2, "text": "고유값은 중요합니다."},
                {"segment_id": "t2", "start": 3, "end": 4, "text": "고유값을 기억하세요."},
            ],
            "board_appearances": [], "board_page_ids": [], "reasons": [],
        }],
    }
    curated = learning_analysis._curate_concepts_with_ai(
        {"transcript": {"segments": []}, "ocr": {"pages": []}},
        result,
    )
    assert curated["ai_curation"]["status"] == "not_configured"
    assert curated["ai_curation"]["provider"] is None
    assert curated["concepts"][0]["name"] == "고유값"



def test_ai_selection_uses_candidate_ids_when_term_spelling_differs():
    from lecture2doc.learning_analysis import _apply_ai_selection

    result = {
        "pages": [{
            "concepts": [{"name": "미분방정식", "importance_score": 75}],
        }],
        "concepts": [{
            "id": "c-board-001",
            "name": "미분 방정식",
            "importance_score": 75,
            "speech_occurrences": [
                {"segment_id": "t1", "start": 12.0, "end": 15.0, "text": "미분 방정식의 분류를 설명하겠습니다."}
            ],
            "board_appearances": [{"board_page_id": "p1", "start": 10.0, "end": 20.0}],
            "board_page_ids": ["p1"],
            "reasons": ["음성에서 1회 언급"],
        }],
    }

    curated = _apply_ai_selection(
        result,
        {
            "topic": "미분 방정식 분류",
            "concepts": [{
                "name": "미분 방정식의 분류",
                "candidate_ids": ["c-board-001"],
                "source_terms": ["미분방정식의 분류"],
                "reason": "강의에서 분류 방법을 설명하는 핵심 내용입니다.",
                "relevance": 96,
            }],
        },
        "qwen3-vl:4b",
    )

    assert len(curated["concepts"]) == 1
    assert curated["concepts"][0]["speech_occurrences"][0]["segment_id"] == "t1"
    assert curated["concepts"][0]["board_page_ids"] == ["p1"]
    assert curated["concepts"][0]["source_terms"] == ["미분 방정식"]
    assert curated["ai_curation"]["selected_count"] == 1
