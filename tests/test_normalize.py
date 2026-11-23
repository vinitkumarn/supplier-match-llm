from suppmatch.normalize import (
    bank_identity,
    clean_text,
    embedding_text,
    normalize_address,
    normalize_id,
    normalize_name,
    normalize_tax_id,
    sap_name,
)


def test_suffix_and_punctuation():
    assert normalize_name("Acme Industries, Inc.") == normalize_name("ACME INDUSTRIES INCORPORATED")
    assert normalize_name("Acme Industries, Inc.") == "ACME INDUSTRIES"


def test_suffix_chain_removed_but_never_empty():
    assert normalize_name("Deccan Foods Pvt. Ltd.") == "DECCAN FOODS"
    assert normalize_name("Kestrel G.m.b.H.") == "KESTREL"
    assert normalize_name("Limited") == "LIMITED"   # a name is never reduced to nothing


def test_suffix_only_at_the_end():
    assert normalize_name("Co Op Farms") == "CO OP FARMS"


def test_street_words():
    assert normalize_address("Main Street") == "MAIN ST"
    assert normalize_address("12 Lake Avenue, Suite 200") == "12 LAKE AVE STE 200"
    assert normalize_address("12 Lake Ave., Ste 200") == "12 LAKE AVE STE 200"
    assert normalize_address("5 Mill Road") == "5 MILL RD"


def test_ids_plain_characters():
    assert normalize_id("12-3456789") == normalize_id("123456789")
    assert normalize_id("GB 123 4567 89") == "GB123456789"
    assert normalize_id("hdfc0001234") == "HDFC0001234"


def test_ids_leading_zeros():
    assert normalize_id("00000012345678") == normalize_id("12345678")
    assert normalize_id("") == "" and normalize_id(None) == ""


def test_tax_id_own_country_prefix_dropped():
    assert normalize_tax_id("GB 123456789", "GB") == normalize_tax_id("123456789", "GB")
    assert normalize_tax_id("DE-123456789", "DE") == normalize_tax_id("123456789", "de")
    assert normalize_tax_id("12-3456789", "US") == "123456789"


def test_tax_id_other_country_prefix_kept():
    assert normalize_tax_id("GB123456789", "DE") == "GB123456789"
    assert normalize_tax_id("GB123456789", "DE") != normalize_tax_id("123456789", "DE")


def test_bank_identity_includes_bank_key():
    a = bank_identity("GB", "20-00-00", "12345678")
    b = bank_identity("gb", "200000", "0000012345678")
    c = bank_identity("GB", "40-11-22", "12345678")
    assert a == b
    assert a != c                       # same account number, different bank
    assert bank_identity("GB", "200000", "") is None


def test_homoglyphs_not_hidden():
    assert clean_text("Аpex") != clean_text("Apex")


def test_sap_name_and_embedding_text():
    assert sap_name("Acme Industries", "") == "Acme Industries"
    assert sap_name("Acme", " Industries ") == "Acme Industries"
    assert embedding_text("Acme Industries, Inc.", "1 Main Street", "Chicago", "us") == (
        "ACME INDUSTRIES | 1 MAIN ST | CHICAGO | US"
    )
